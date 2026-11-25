"""Aprendizaje no supervisado: segmentación (KMeans) y anomalías (IsolationForest).

* Los clústeres se ajustan SOLO con variables conocidas antes/al momento de la
  venta (`features.make_features`), igual que los modelos supervisados; no se
  usa el resultado económico para formar los grupos (sería fuga de información).
* El resultado económico y la tasa de devolución se añaden después, únicamente
  para **interpretar** cada segmento.
* `contamination=0.05` en IsolationForest: se marca el 5 % más atípico. No se
  usa `"auto"` porque en estos datos marcaba más de la mitad de las ventas y
  dejaba la alerta sin utilidad operativa.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from src import features as feat
from src import multivariate as mv

MODEL_PATH = Path("models/unsupervised.joblib")
K_MIN, K_MAX = 2, 8


def _select_best_k(X_trans: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX,
                   random_state: int = 42) -> tuple[int, dict, dict]:
    """Elige k por silueta (mayor es mejor); desempata con el k más pequeño.

    Devuelve (best_k, siluetas, inercias) para poder mostrar también el codo.
    """
    n = X_trans.shape[0]
    upper = min(k_max, n - 1)
    if upper < k_min:
        return 1, {}, {}

    sil, inertia = {}, {}
    best_k, best_score = k_min, -np.inf
    for k in range(k_min, upper + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_trans)
        inertia[k] = round(float(km.inertia_), 4)
        if len(set(labels)) > 1:
            score = float(silhouette_score(X_trans, labels))
            sil[k] = round(score, 4)
            if score > best_score:
                best_score, best_k = score, k
    return best_k, sil, inertia


def _outcome_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Medias observadas por clúster: descriptivas, no usadas para entrenar."""
    cols = [c for c in ["total", "ingresos_productos", "costos_envio",
                        "cargo_venta"] if c in df]
    out = pd.DataFrame({"cluster": labels})
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    # Magnitudes: el signo negativo de ML es contable, no económico.
    for c in ("costos_envio", "cargo_venta"):
        if c in out:
            out[c] = out[c].abs()

    # Tasa de devolución (solo si el objetivo verificable existe). Import perezoso
    # para no acoplar el módulo al modelo supervisado en tiempo de importación.
    try:
        from src.returns_model import build_target

        y = build_target(df)
        out["tasa_devolucion"] = y.to_numpy()
    except Exception:
        pass

    g = out.groupby("cluster", as_index=False).mean(numeric_only=True)
    return g.round(3)


def fit_unsupervised(df: pd.DataFrame, n_clusters: int | None = None,
                     random_state: int = 42) -> dict:
    """Entrena KMeans + IsolationForest + PCA de apoyo sobre features pre-venta."""
    X_raw, num_cols, cat_cols = feat.make_features(df)
    if len(X_raw) < 30:
        return {"viable": False,
                "razon": "Menos de 30 filas: el aprendizaje no supervisado no es fiable."}

    prep = feat.preprocessor(num_cols, cat_cols)
    X_trans = prep.fit_transform(X_raw)

    if n_clusters is None:
        best_k, sil_scores, inertia = _select_best_k(X_trans, random_state=random_state)
    else:
        best_k = max(2, min(int(n_clusters), len(X_raw) - 1))
        km_tmp = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
        labels_tmp = km_tmp.fit_predict(X_trans)
        sil_scores = ({best_k: round(float(silhouette_score(X_trans, labels_tmp)), 4)}
                      if len(set(labels_tmp)) > 1 else {})
        inertia = {best_k: round(float(km_tmp.inertia_), 4)}

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_trans)

    iso = IsolationForest(contamination=0.05, random_state=random_state)
    iso_pred = iso.fit_predict(X_trans)          # 1 = normal, -1 = anomalía
    anomaly_score = -iso.score_samples(X_trans)  # mayor = más anómalo

    # PCA de 2 componentes para visualizar la estructura multivariada.
    pca2 = PCA(n_components=2, random_state=random_state)
    coords = pca2.fit_transform(X_trans)
    var2 = [round(float(v), 4) for v in pca2.explained_variance_ratio_]

    # Perfil de segmentos: medias de las variables de entrada + resultados.
    perfil = X_raw.copy()
    perfil["cluster"] = labels
    perfil["es_anomalia"] = iso_pred == -1
    perfil["pca_1"], perfil["pca_2"] = coords[:, 0], coords[:, 1]
    agg = {c: "mean" for c in ["unidades", "precio_unitario", "ingreso_esperado"] if c in perfil}
    agg.update({"cluster": "count", "es_anomalia": "mean"})
    base = perfil.groupby("cluster", as_index=False).agg(agg).rename(
        columns={"cluster": "n_registros", "es_anomalia": "tasa_anomalias"})
    base.insert(0, "cluster", range(len(base)))
    resumen = base.merge(_outcome_profile(df, labels), on="cluster", how="left")
    resumen = resumen.rename(columns={"total": "resultado_ml_medio",
                                      "ingresos_productos": "ingreso_medio"})

    anomalias = pd.DataFrame({
        "cluster": labels,
        "score_anomalia": np.round(anomaly_score, 4),
    }, index=df.index)
    for c in ("id_venta", "sku", "transportista", "fecha_venta"):
        if c in df:
            anomalias[c] = df[c]
    anomalias = anomalias[iso_pred == -1].sort_values("score_anomalia", ascending=False)

    bundle = {
        "nombre": f"KMeans (k={best_k}) + IsolationForest",
        "preprocessor": prep,
        "kmeans": kmeans,
        "isolation_forest": iso,
        "pca": pca2,
        "feature_cols": list(X_raw.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "inertia": inertia,
        "varianza_pca2": var2,
    }

    return {
        "viable": True,
        "mejor": bundle["nombre"],
        "n_registros": int(len(X_raw)),
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "inertia": inertia,
        "silueta": sil_scores.get(best_k),
        "n_anomalias": int((iso_pred == -1).sum()),
        "pct_anomalias": round(float((iso_pred == -1).mean() * 100), 2),
        "resumen_clusters": resumen,
        "perfil": perfil,
        "anomalias": anomalias,
        "pca_2d": perfil[["pca_1", "pca_2"]].assign(cluster=labels),
        "varianza_pca2": var2,
        "multivariado": mv.multivariate_report(df),
        "labels": labels,
        "is_anomaly": iso_pred == -1,
        "anomaly_score": anomaly_score,
        "bundle": bundle,
    }


def save_model(result: dict, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = dict(result["bundle"])
    # Se guarda un resumen liviano para poder recuperar la vista desde disco.
    bundle["report"] = {
        "n_registros": result.get("n_registros"),
        "best_k": result.get("best_k"),
        "silhouette_scores": result.get("silhouette_scores"),
        "inertia": result.get("inertia"),
        "n_anomalias": result.get("n_anomalias"),
        "pct_anomalias": result.get("pct_anomalias"),
        "resumen_clusters": result.get("resumen_clusters"),
        "varianza_pca2": result.get("varianza_pca2"),
        "multivariado": result.get("multivariado"),
        "anomalias": result.get("anomalias"),
        "pca_2d": result.get("pca_2d"),
    }
    joblib.dump(bundle, path)
    return path


def predict_unsupervised(df: pd.DataFrame, path: Path = MODEL_PATH) -> pd.DataFrame:
    """Asigna clúster, coordenadas PCA y puntuación de anomalía a datos nuevos."""
    bundle = joblib.load(path)
    X_raw, _, _ = feat.make_features(df)
    X_raw = X_raw.reindex(columns=bundle["feature_cols"])
    X_trans = bundle["preprocessor"].transform(X_raw)
    out = pd.DataFrame(index=df.index)
    out["cluster"] = bundle["kmeans"].predict(X_trans)
    iso_pred = bundle["isolation_forest"].predict(X_trans)
    out["es_anomalia"] = iso_pred == -1
    out["score_anomalia"] = -bundle["isolation_forest"].score_samples(X_trans)
    coords = bundle["pca"].transform(X_trans)
    out["pca_1"], out["pca_2"] = coords[:, 0], coords[:, 1]
    return out


# Alias consistente con profitability_model / returns_model.
predict = predict_unsupervised
