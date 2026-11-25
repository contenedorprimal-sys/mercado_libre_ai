"""Análisis multivariado exploratorio (correlación, VIF y PCA).

No es inferencia multivariada formal (MANOVA/Hotelling): el conjunto es un
registro censal de ventas, no una muestra aleatoria, y las variables están
fuertemente correlacionadas por construcción (precio × unidades = ingreso).
Por eso se usa como herramienta **descriptiva** para:

* detectar redundancia entre variables (correlación y VIF),
* resumir la estructura común en pocos componentes (PCA),
* justificar o descartar la segmentación (clustering) sobre esas variables.

Todas las funciones son puras y no dependen de Streamlit, de modo que se
pueden probar sin interfaz ni base de datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Variables numéricas con significado económico/operativo. Se incluyen solo si
# existen en el DataFrame; `ingreso_esperado` se deriva cuando es posible.
CANDIDATAS = ["unidades", "precio_unitario", "ingreso_esperado",
              "mes", "dia_semana", "costos_envio", "cargo_venta",
              "descuentos", "anulaciones", "total", "resultado_economico"]
MIN_FILAS = 30
MIN_COLUMNAS = 3


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y limpia las variables numéricas del análisis multivariado."""
    d = df.copy()
    if {"unidades", "precio_unitario"} <= set(d) and "ingreso_esperado" not in d:
        d["ingreso_esperado"] = d["unidades"] * d["precio_unitario"]
    cols = [c for c in CANDIDATAS if c in d]
    if len(cols) < MIN_COLUMNAS:
        return pd.DataFrame(index=d.index)
    x = d[cols].apply(pd.to_numeric, errors="coerce")
    # `costos_envio` y `cargo_venta` vienen con signo negativo; se usa magnitud
    # para que la interpretación sea "cuánto cuesta", no "cuánto resta".
    for c in ("costos_envio", "cargo_venta"):
        if c in x:
            x[c] = x[c].abs()
    return x.replace([np.inf, -np.inf], np.nan)


def _clean_numeric(x: pd.DataFrame) -> pd.DataFrame:
    """Quita columnas constantes/vacías e imputa con la mediana.

    Sin esto, una columna como `descuentos` (todo ceros en algunos periodos)
    vuelve singular la matriz de correlación y arruina VIF y PCA.
    """
    x = x.copy()
    suficientes = [c for c in x if x[c].notna().sum() >= MIN_FILAS and x[c].nunique(dropna=True) > 1]
    x = x[suficientes]
    if not len(suficientes):
        return x
    return x.fillna(x.median(numeric_only=True))


def correlation_matrix(x: pd.DataFrame) -> pd.DataFrame:
    return _clean_numeric(x).corr(numeric_only=True).round(3)


def vif_table(x: pd.DataFrame) -> pd.DataFrame:
    """VIF a partir de la inversa de la matriz de correlación.

    VIF > 10 indica redundancia severa; VIF > 5, redundancia moderada.
    Se calcula sin statsmodels para no añadir dependencias.
    """
    x = _clean_numeric(x)
    if len(x) < MIN_FILAS or x.shape[1] < 2:
        return pd.DataFrame(columns=["variable", "VIF", "diagnostico"])
    corr = x.corr().to_numpy()
    # Regularización mínima para matrices singulares (columnas colineales).
    corr = corr + np.eye(corr.shape[0]) * 1e-6
    try:
        inv = np.linalg.inv(corr)
        vif = np.diag(inv)
    except np.linalg.LinAlgError:
        return pd.DataFrame(columns=["variable", "VIF", "diagnostico"])
    out = pd.DataFrame({"variable": x.columns, "VIF": np.round(vif, 2)})
    out["diagnostico"] = pd.cut(
        out["VIF"],
        bins=[-np.inf, 5, 10, np.inf],
        labels=["Aceptable", "Moderada", "Severa"],
    ).astype(str)
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def pca_summary(x: pd.DataFrame, n_components: int | None = None) -> dict:
    """PCA sobre variables estandarizadas: varianza explicada y cargas."""
    x = _clean_numeric(x)
    if len(x) < MIN_FILAS or x.shape[1] < 2:
        return {"aplica": False,
                "razon": f"Se requieren ≥ {MIN_FILAS} filas completas y ≥ 2 variables."}
    n = min(n_components or x.shape[1], x.shape[1], len(x))
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    pca = PCA(n_components=n, random_state=42)
    scores = pca.fit_transform(z)
    var = pca.explained_variance_ratio_
    cargas = pd.DataFrame(
        pca.components_[:min(3, n)].T,
        index=x.columns,
        columns=[f"PC{i + 1}" for i in range(min(3, n))],
    ).round(3)
    return {
        "aplica": True,
        "n": int(len(x)),
        "varianza_explicada": [round(float(v), 4) for v in var],
        "varianza_acumulada": [round(float(v), 4) for v in np.cumsum(var)],
        "n_componentes_80": int(np.searchsorted(np.cumsum(var), 0.80) + 1),
        "cargas": cargas,
        "scores": pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"], index=x.index),
        "scaler": scaler,
        "pca": pca,
    }


def multivariate_report(df: pd.DataFrame, n_components: int | None = None) -> dict:
    """Informe completo y decisión de aplicabilidad para el proyecto."""
    x = numeric_frame(df)
    if len(x) < MIN_FILAS or x.shape[1] < MIN_COLUMNAS:
        return {
            "aplica": False,
            "razon": (f"Se necesitan ≥ {MIN_FILAS} filas y ≥ {MIN_COLUMNAS} variables "
                      f"numéricas; hay {len(x)} filas y {x.shape[1]} variables."),
        }
    pca = pca_summary(x, n_components)
    return {
        "aplica": True,
        "variables": list(x.columns),
        "n": int(len(x)),
        "correlacion": correlation_matrix(x),
        "vif": vif_table(x),
        "pca": pca,
        "razon": (
            "Aplica como análisis multivariado EXPLORATORIO: varias variables "
            "económicas por venta, correlacionadas, que se resumen con PCA y se "
            "agrupan con clustering. No se usa inferencia (MANOVA) porque los "
            "datos son un registro completo, no una muestra aleatoria."
        ),
        "colinealidad": (
            "VIF alto (>10) confirma redundancia: p. ej. ingreso_esperado y "
            "cargo_venta son casi la misma variable. Para modelar no se deben "
            "usar juntas; para segmentar sirve reducir con PCA."
        ),
    }
