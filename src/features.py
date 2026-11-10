"""Variables disponibles ANTES de la venta/devolución (evita data leakage)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["unidades", "precio_unitario", "ingreso_esperado", "mes", "dia_semana"]
CAT = ["sku", "tipo_publicacion", "venta_publicidad", "forma_de_entrega", "transportista"]


def make_features(d: pd.DataFrame):
    x = pd.DataFrame(index=d.index)
    for c in ["unidades", "precio_unitario"]:
        if c in d:
            x[c] = d[c]
    if {"unidades", "precio_unitario"} <= set(x):
        x["ingreso_esperado"] = x["unidades"] * x["precio_unitario"]
    if "fecha_venta" in d:
        x["mes"] = d["fecha_venta"].dt.month
        x["dia_semana"] = d["fecha_venta"].dt.dayofweek
    for c in CAT:
        if c in d:
            x[c] = d[c].astype(str).str.strip().replace({"nan": "NA", "None": "NA", "": "NA"})
    num = [c for c in NUM if c in x]
    cat = [c for c in CAT if c in x]
    return x[num + cat], num, cat


def preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=3,
                                               sparse_output=False))]), cat),
    ])
