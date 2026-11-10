"""Carga, limpieza y calidad de datos del reporte de ventas de Mercado Libre.

El archivo original NUNCA se modifica: todo se hace sobre copias en memoria.
"""
import re
import unicodedata
import numpy as np
import pandas as pd

ALIASES = {
    "de_venta": "id_venta", "fecha_de_venta": "fecha_venta",
    "descripcion_del_estado": "descripcion_estado",
    "ingresos_por_productos": "ingresos_productos",
    "cargo_por_venta_e_impuestos": "cargo_venta",
    "ingresos_por_envio": "ingresos_envio", "costos_de_envio": "costos_envio",
    "costo_de_envio_basado_en_medidas_y_peso_declarados": "costo_envio_declarado",
    "cargo_por_diferencias_en_medidas_y_peso_del_paquete": "cargo_diferencias",
    "descuentos_y_bonificaciones": "descuentos", "anulaciones_y_reembolsos": "anulaciones",
    "venta_por_publicidad": "venta_publicidad", "de_publicacion": "id_publicacion",
    "titulo_de_la_publicacion": "titulo",
    "precio_unitario_de_venta_de_la_publicacion": "precio_unitario",
    "tipo_de_publicacion": "tipo_publicacion",
}
MONEY = ["ingresos_productos", "cargo_venta", "ingresos_envio", "costos_envio",
         "costo_envio_declarado", "cargo_diferencias", "descuentos", "anulaciones",
         "total", "precio_unitario"]
DATES = ["fecha_venta", "fecha_en_camino", "fecha_entregado",
         "devoluciones_fecha_en_camino", "devoluciones_fecha_entregado", "fecha_de_revision"]
# Datos personales / sensibles: se excluyen siempre del conjunto de modelado.
PII_BASES = {"factura_adjunta", "datos_personales_o_de_empresa", "tipo_y_numero_de_documento",
             "direccion", "tipo_de_contribuyente", "cfdi", "tipo_de_usuario", "regimen_fiscal",
             "comprador", "negocio", "ife", "domicilio", "municipio_alcaldia", "codigo_postal",
             "numero_de_seguimiento", "url_de_seguimiento", "titulo"}
MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
         "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11,
         "diciembre": 12}


def slug(text) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"\(.*?\)", "", t)
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def _build_columns(headers, sections):
    """Nombres únicos. Las columnas repetidas (Estado, Unidades, Forma de entrega...)
    se desambiguan con el nombre de su sección: la 1a conserva el nombre simple."""
    seen, names = set(), []
    for h, s in zip(headers, sections):
        base = slug(h)
        name = ALIASES.get(base, base)
        if name in seen and s:
            name = f"{slug(s)}_{base}"
        root, i = name, 2
        while name in seen:
            name = f"{root}_{i}"
            i += 1
        seen.add(name)
        names.append(name)
    return names


def load_file(file) -> pd.DataFrame:
    """Lee .xlsx/.xls/.csv (ruta u objeto subido). Detecta la fila de encabezados
    (la que contiene '# de venta') y la fila de secciones (Ventas, Envíos...)."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file, header=None, dtype=object)
    elif name.endswith(".csv"):
        raw = pd.read_csv(file, header=None, dtype=object, encoding="utf-8-sig")
    else:
        raise ValueError("Formato no soportado: usa .xlsx, .xls o .csv")
    hdr = next((i for i in range(min(len(raw), 30))
                if any(slug(c) == "de_venta" for c in raw.iloc[i].dropna())), None)
    if hdr is None:
        raise ValueError("No se encontró la columna '# de venta'. ¿Es un reporte de ventas de Mercado Libre?")
    sections = [""] * raw.shape[1]
    if hdr > 0:
        row = raw.iloc[hdr - 1]
        sec = row.where(row.notna()).ffill()
        sections = [("" if pd.isna(x) else str(x)) for x in sec]
    headers = [("" if pd.isna(h) else h) for h in raw.iloc[hdr]]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = _build_columns(headers, sections)
    df = df.loc[:, [c for c in df.columns if c]]
    return df.dropna(how="all").reset_index(drop=True)


def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.replace(r"[\$\s]", "", regex=True).str.replace(",", "", regex=False)
    t = t.replace({"nan": np.nan, "None": np.nan, "": np.nan, "-": np.nan})
    return pd.to_numeric(t, errors="coerce")


def parse_date_es(v):
    """Acepta datetime o textos como '15 de enero de 2025 10:23 hs.'."""
    if pd.isna(v):
        return pd.NaT
    if hasattr(v, "year") and not isinstance(v, str):
        return pd.Timestamp(v)
    m = re.search(r"(\d{1,2}) de (\w+) de (\d{4})(?:\s+(\d{1,2}):(\d{2}))?", str(v).lower())
    if m and m.group(2) in MESES:
        d, mes, y, h, mi = m.groups()
        return pd.Timestamp(int(y), MESES[mes], int(d), int(h or 0), int(mi or 0))
    if re.match(r"^\d{4}-\d{2}-\d{2}", str(v)):
        return pd.to_datetime(v, errors="coerce")  # ISO: nunca dayfirst
    return pd.to_datetime(v, errors="coerce", dayfirst=True)


def clean(df: pd.DataFrame):
    """Devuelve (df_limpio, log_de_decisiones). Elimina datos personales."""
    log, d = [], df.copy()
    for c in MONEY:
        if c in d:
            d[c] = to_number(d[c])
    for c in d.columns:
        if c.startswith("unidades") or c.startswith("devoluciones_unidades") or c == "dinero_a_favor":
            d[c] = to_number(d[c])
    for c in DATES:
        if c in d:
            d[c] = pd.to_datetime(d[c].map(parse_date_es))
    n0 = len(d)
    d = d.drop_duplicates().reset_index(drop=True)
    if n0 - len(d):
        log.append(f"Se eliminaron {n0 - len(d)} filas completamente duplicadas.")
    pii = [c for c in d.columns if any(c == b or c.endswith("_" + b) for b in PII_BASES)]
    d = d.drop(columns=pii)
    log.append(f"Columnas con datos personales excluidas del modelado: {len(pii)}.")
    if "fecha_venta" in d:
        log.append(f"{int(d['fecha_venta'].isna().sum())} filas sin fecha de venta válida (se excluyen al entrenar).")
    log.append("Importes: texto -> número (se quitan $ y comas). Los importes faltantes NO se "
               "rellenan con ceros; la imputación ocurre solo dentro del pipeline de ML.")
    return d, log


def quality_report(d: pd.DataFrame) -> dict:
    rep = {"filas": len(d), "columnas": d.shape[1]}
    rep["faltantes_pct"] = (d.isna().mean() * 100).round(1).sort_values(ascending=False)
    if "id_venta" in d:
        rep["id_venta_repetidos"] = int(d["id_venta"].duplicated().sum())
    out = {}
    for c in [c for c in MONEY if c in d]:
        x = d[c].dropna()
        if len(x) > 10:
            q1, q3 = x.quantile([.25, .75])
            iqr = q3 - q1
            out[c] = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
    rep["atipicos_iqr3"] = out
    inc = {}
    if "unidades" in d:
        inc["unidades <= 0"] = int((d["unidades"] <= 0).sum())
    if {"fecha_en_camino", "fecha_entregado"} <= set(d):
        inc["entrega anterior a envío"] = int((d["fecha_entregado"] < d["fecha_en_camino"]).sum())
    comp = [c for c in ["ingresos_productos", "cargo_venta", "ingresos_envio", "costos_envio",
                        "cargo_diferencias", "descuentos", "anulaciones"] if c in d]
    if "total" in d and comp:
        diff = (d[comp].fillna(0).sum(axis=1) - d["total"]).abs()
        inc["Total != suma de componentes (>1 MXN)"] = int((diff > 1).sum())
    rep["inconsistencias"] = inc
    return rep
