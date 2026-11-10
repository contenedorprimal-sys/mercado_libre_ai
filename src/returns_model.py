"""Riesgo de devolución (clasificación).

Objetivo (solo con evidencia verificable):
  1 = hay evidencia de devolución (envío de devolución, unidades devueltas o estado 'devol*')
  0 = entregada hace >= 30 días, sin devolución, sin reclamo y sin datos en el bloque Devoluciones
  NaN = ambiguo (cancelada, reclamo sin devolución, entrega reciente...) -> EXCLUIDO.
Se usan solo variables previas al evento; nada de motivo/resultado/fechas de cierre.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .evaluation import classification_metrics, importance_table, temporal_split
from .features import make_features, preprocessor

MIN_POS, MIN_NEG, MIN_POS_TEST, VENTANA_DIAS = 30, 100, 5, 30


def _txt(d, c):
    return d[c].astype(str).str.lower() if c in d else pd.Series("", index=d.index)


def _any_filled(d, cols):
    cols = [c for c in cols if c in d]
    return d[cols].notna().any(axis=1) if cols else pd.Series(False, index=d.index)


def build_target(df: pd.DataFrame) -> pd.Series:
    unid_dev = [c for c in df.columns if c.startswith("devoluciones_unidades")]
    dev_cols = [c for c in df.columns if c.startswith("devoluciones_")] + \
               [c for c in ["revisado_por_mercado_libre", "dinero_a_favor", "resultado", "destino",
                            "motivo_del_resultado"] if c in df]
    texto = _txt(df, "estado") + " " + _txt(df, "descripcion_estado")
    units_pos = (df[unid_dev].fillna(0).sum(axis=1) > 0) if unid_dev else pd.Series(False, index=df.index)
    pos = (_any_filled(df, ["devoluciones_fecha_en_camino", "devoluciones_fecha_entregado"])
           | units_pos | texto.str.contains("devol"))
    reclamo = _any_filled(df, ["reclamo_abierto", "reclamo_cerrado", "con_mediacion"])
    ref = df["fecha_venta"].max()
    f_ent = df["fecha_entregado"] if "fecha_entregado" in df else pd.Series(pd.NaT, index=df.index)
    maduro = f_ent.notna() & (f_ent <= ref - pd.Timedelta(days=VENTANA_DIAS))
    neg = (texto.str.contains("entreg") & ~texto.str.contains("devol|cancel")
           & ~reclamo & ~_any_filled(df, dev_cols) & maduro & ~pos)
    y = pd.Series(np.nan, index=df.index)
    y[neg] = 0
    y[pos] = 1
    return y


def diagnose(df: pd.DataFrame) -> dict:
    y = build_target(df)
    return {"ventas_totales": len(df), "devoluciones(1)": int((y == 1).sum()),
            "no_devoluciones(0)": int((y == 0).sum()), "ambiguas(excluidas)": int(y.isna().sum())}


def _models(pos_weight):
    m = {"Baseline (prior)": DummyClassifier(strategy="prior"),
         "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
         "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                                 class_weight="balanced", random_state=42),
         "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                 class_weight="balanced_subsample",
                                                 random_state=42, n_jobs=-1)}
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, scale_pos_weight=pos_weight,
                                     eval_metric="logloss", random_state=42)
    except ImportError:
        pass
    return m


def train(df: pd.DataFrame, models_dir="models") -> dict:
    d = df.copy()
    d["devolucion"] = build_target(d)
    diag = diagnose(df)
    d = d.dropna(subset=["devolucion", "fecha_venta"])
    d["devolucion"] = d["devolucion"].astype(int)
    npos, nneg = int(d.devolucion.sum()), int((d.devolucion == 0).sum())
    if npos < MIN_POS or nneg < MIN_NEG:
        return {"viable": False, "diagnostico": diag,
                "razon": f"Datos insuficientes: {npos} devoluciones y {nneg} no-devoluciones "
                         f"verificables (mínimo {MIN_POS} y {MIN_NEG}). No se entrena el modelo; "
                         "se recomienda análisis descriptivo o más historial."}
    tr, te = temporal_split(d)
    if te.devolucion.sum() < MIN_POS_TEST or tr.devolucion.nunique() < 2:
        return {"viable": False, "diagnostico": diag,
                "razon": "El periodo de prueba tiene muy pocas devoluciones para evaluar de forma confiable."}
    Xtr, num, cat = make_features(tr)
    Xte, _, _ = make_features(te)
    ytr, yte = tr.devolucion, te.devolucion
    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    rows, fitted = [], {}
    for name, est in _models(pw).items():
        pipe = Pipeline([("prep", preprocessor(num, cat)), ("model", est)]).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        rows.append({"Modelo": name, **classification_metrics(yte, (proba >= .5).astype(int), proba)})
        fitted[name] = pipe
    metrics = pd.DataFrame(rows)
    cand = metrics[~metrics["Modelo"].str.startswith("Baseline")].sort_values("ROC-AUC", ascending=False)
    best = cand["Modelo"].iloc[0]
    proba = fitted[best].predict_proba(Xte)[:, 1]
    umbrales = {"medio": float(np.quantile(proba, .5)), "alto": float(np.quantile(proba, .8))}
    imp = importance_table(fitted[best], Xte, yte, "roc_auc")
    Path(models_dir).mkdir(exist_ok=True)
    joblib.dump({"pipeline": fitted[best], "nombre": best, "umbrales": umbrales,
                 "metrics": metrics}, Path(models_dir) / "returns.joblib")
    return {"viable": True, "diagnostico": diag, "metrics": metrics, "mejor": best,
            "importancia": imp, "umbrales": umbrales, "n_train": len(tr), "n_test": len(te),
            "tasa_train": float(ytr.mean()), "tasa_test": float(yte.mean())}


def risk_band(p, umbrales):
    """Riesgo RELATIVO (probabilidades con class_weight no están calibradas)."""
    return np.where(p >= umbrales["alto"], "Alto", np.where(p >= umbrales["medio"], "Medio", "Bajo"))


def predict(df: pd.DataFrame, models_dir="models") -> pd.DataFrame:
    b = joblib.load(Path(models_dir) / "returns.joblib")
    X, _, _ = make_features(df)
    p = b["pipeline"].predict_proba(X)[:, 1]
    return pd.DataFrame({"prob_devolucion": p, "riesgo": risk_band(p, b["umbrales"])}, index=df.index)
