"""Rentabilidad: cálculo económico + modelos de regresión comparados.

Resultado económico por venta = Total de Mercado Libre (ya incluye ingresos, cargos,
envíos, descuentos y reembolsos; así NO se contabilizan dos veces) - costo de adquisición.
Sin catálogo de costos el resultado es PARCIAL y así se etiqueta.

Variable a predecir: resultado económico por venta. Predictores: solo información
conocida antes de la venta (precio, unidades, SKU, tipo de publicación, envío, mes).
NO se usan ingresos/cargos/costos/reembolsos reales (serían fuga de información).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .evaluation import importance_table, regression_metrics, temporal_split
from .features import make_features, preprocessor
from .selection import choose_model

MIN_ROWS = 100
COMPONENTES = ["ingresos_productos", "ingresos_envio", "cargo_venta", "costos_envio",
               "cargo_diferencias", "descuentos", "anulaciones"]


def add_economics(df: pd.DataFrame, costs=None) -> pd.DataFrame:
    """costs: DataFrame con columnas 'sku' y 'costo_unitario'."""
    d = df.copy()
    comp = [c for c in COMPONENTES if c in d]
    if "total" in d and d["total"].notna().any():
        d["resultado_ml"] = d["total"]
    else:
        d["resultado_ml"] = d[comp].fillna(0).sum(axis=1)
    d["costo_adquisicion"] = np.nan
    if costs is not None and "sku" in d:
        c = costs.rename(columns=str.lower).drop_duplicates("sku").set_index("sku")["costo_unitario"]
        d["costo_adquisicion"] = d["sku"].map(c) * (d["unidades"] if "unidades" in d else 1)
    d["resultado_economico"] = d["resultado_ml"] - d["costo_adquisicion"].fillna(0)
    d["tipo_resultado"] = np.where(d["costo_adquisicion"].notna(), "estimado con costo",
                                   "PARCIAL (sin costo de adquisición)")
    return d


def _models():
    m = {"Baseline (media)": DummyRegressor(),
         "Linear Regression": LinearRegression(),
         "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                random_state=42, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
    except ImportError:
        pass
    return m


def train(df: pd.DataFrame, costs=None, models_dir="models") -> dict:
    d = add_economics(df, costs).dropna(subset=["fecha_venta", "resultado_economico"])
    if len(d) < MIN_ROWS:
        return {"viable": False, "razon": f"Solo {len(d)} ventas utilizables; se requieren ≥ {MIN_ROWS}."}
    tr, te = temporal_split(d)
    Xtr, num, cat = make_features(tr)
    Xte, _, _ = make_features(te)
    ytr, yte = tr["resultado_economico"], te["resultado_economico"]
    rows, fitted, errores = [], {}, {}
    for name, est in _models().items():
        try:
            pipe = Pipeline([("prep", preprocessor(num, cat)), ("model", est)]).fit(Xtr, ytr)
        except Exception as exc:          # una librería opcional no debe tumbar el resto
            errores[name] = str(exc)
            continue
        rows.append({"Modelo": name, **regression_metrics(yte, pipe.predict(Xte))})
        fitted[name] = pipe
    if not fitted:
        return {"viable": False, "razon": "Ningún modelo pudo entrenarse.", "errores": errores}
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    seleccion = choose_model(metrics, "regression")
    best = seleccion["best"] or metrics["Modelo"].iloc[0]
    imp = importance_table(fitted[best], Xte, yte, "neg_root_mean_squared_error")
    pred = te[["fecha_venta"] + [c for c in ["sku"] if c in te]].copy()
    pred["real"], pred["predicho"] = yte.values, fitted[best].predict(Xte)
    Path(models_dir).mkdir(exist_ok=True)
    joblib.dump({"pipeline": fitted[best], "nombre": best, "metrics": metrics,
                 "seleccion": seleccion,
                 "tipo_resultado": d["tipo_resultado"].value_counts().to_dict()},
                Path(models_dir) / "profitability.joblib")
    return {"viable": True, "metrics": metrics, "mejor": best, "importancia": imp,
            "predicciones": pred, "n_train": len(tr), "n_test": len(te),
            "periodo_train": (tr.fecha_venta.min(), tr.fecha_venta.max()),
            "periodo_test": (te.fecha_venta.min(), te.fecha_venta.max()),
            "seleccion": seleccion, "confiable": seleccion["confiable"],
            "advertencia": None if seleccion["confiable"] else seleccion["reason"],
            "errores": errores,
            "tipo_resultado": d["tipo_resultado"].value_counts().to_dict()}
