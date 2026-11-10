"""Interfaz Streamlit — Predicción de rentabilidad y riesgo de devoluciones (Mercado Libre).

Ejecutar en local:  streamlit run app.py
En Docker:          docker compose up --build

Persistencia: los datos cargados se guardan en DATA_DIR/current y los modelos
entrenados en MODELS_DIR, de modo que sobreviven a reinicios del contenedor.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data_processing as dp
from src import profitability_model as pm
from src import recommendations as rec
from src import returns_model as rm

# --------------------------------------------------------------------------- #
# Configuración general
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="ML Analytics IA · Mercado Libre",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
CURRENT_DIR = DATA_DIR / "current"
DEMO_FILE = DATA_DIR / "demo_simulado.csv"
MODEL_FILES = {"prof": "profitability.joblib", "ret": "returns.joblib"}

VERDE = "#1E9E5A"
AZUL = "#0B2545"
AZUL_2 = "#13315C"
AMBAR = "#F4A261"
ROJO = "#D1495B"
GRIS = "#6B7A8F"
COLORES = [VERDE, AZUL_2, AMBAR, ROJO, "#457B9D", "#8AB17D", "#E9C46A", "#9B5DE5"]

SECCIONES = [
    "Inicio",
    "Carga de datos",
    "Análisis exploratorio",
    "Rentabilidad",
    "Predicción de devoluciones",
    "Comparación de modelos",
    "Recomendaciones inteligentes",
    "Reportes",
]
ICONOS = {
    "Inicio": "🏠",
    "Carga de datos": "📤",
    "Análisis exploratorio": "📊",
    "Rentabilidad": "💰",
    "Predicción de devoluciones": "🔄",
    "Comparación de modelos": "⚖️",
    "Recomendaciones inteligentes": "💡",
    "Reportes": "📄",
}
S = st.session_state

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
  :root { --verde:#1E9E5A; --azul:#0B2545; --fondo:#F5F8F6; }
  .stApp { background: linear-gradient(180deg,#F5F8F6 0%,#EEF4F0 100%); }
  #MainMenu, footer { visibility: hidden; }
  [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] { display:none !important; }

  /* ---------- barra lateral ---------- */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg,#0B2545 0%,#13315C 65%,#0E3B2E 100%);
      border-right: 1px solid rgba(255,255,255,.08);
  }
  [data-testid="stSidebar"] { color:#EAF2EE; }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#EAF2EE; }
  [data-testid="stSidebar"] .stRadio label {
      border-radius:10px; padding:6px 8px; transition:background .15s;
  }
  [data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.08); }
  [data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] {
      background:rgba(30,158,90,.35);
  }
  .brand { display:flex; gap:.7rem; align-items:center; margin:.2rem 0 1.1rem; }
  .brand .logo {
      width:44px;height:44px;border-radius:12px;display:grid;place-items:center;
      background:linear-gradient(135deg,#1E9E5A,#2EC27E); font-size:1.35rem;
      box-shadow:0 6px 18px rgba(30,158,90,.35);
  }
  .brand b { display:block; font-size:1.05rem; line-height:1.1; color:#fff; }
  .brand span { font-size:.72rem; color:#9FC4B4; letter-spacing:.06em; text-transform:uppercase; }
  .side-card {
      background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.1);
      border-radius:12px; padding:.75rem .85rem; margin:.35rem 0 .6rem; font-size:.8rem;
      line-height:1.5; color:#D7E7DF;
  }
  .side-card b { color:#fff; }
  .dot { display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px; }
  .dot.on { background:#2EC27E; box-shadow:0 0 0 3px rgba(46,194,126,.18); }
  .dot.off { background:#F4A261; }

  /* ---------- encabezado hero ---------- */
  .hero {
      position:relative; overflow:hidden; border-radius:20px; padding:2rem 2.2rem;
      background: radial-gradient(1100px 320px at 12% -20%,rgba(46,194,126,.35),transparent 60%),
                  linear-gradient(120deg,#0B2545 0%,#13315C 55%,#0E3B2E 100%);
      color:#fff; margin-bottom:1.2rem; box-shadow:0 18px 40px rgba(11,37,69,.18);
  }
  .hero:after {
      content:""; position:absolute; right:-60px; top:-70px; width:280px; height:280px;
      border-radius:50%; background:radial-gradient(circle,rgba(30,158,90,.45),transparent 65%);
  }
  .hero .kicker { font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:#8FE3B5; }
  .hero h1 { color:#fff !important; font-size:2.05rem; line-height:1.15; margin:.45rem 0 .6rem; }
  .hero p { color:#C8D9D2; max-width:780px; margin:0 0 1rem; font-size:.95rem; }
  .pill {
      display:inline-block; padding:.3rem .7rem; margin:.15rem .35rem .15rem 0; border-radius:999px;
      background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18);
      font-size:.74rem; color:#EAF2EE;
  }

  /* ---------- títulos ---------- */
  h1,h2,h3 { color:var(--azul) !important; letter-spacing:-.01em; }
  h1 { font-weight:800; }
  .page-head { display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1rem; }
  .page-head .icon {
      width:46px;height:46px;border-radius:13px;display:grid;place-items:center;font-size:1.35rem;
      background:#E4F4EA; border:1px solid #C9E8D5;
  }
  .page-head h1 { margin:0; font-size:1.7rem; }
  .page-head p { margin:0; color:#5A6B7B; font-size:.85rem; }

  /* ---------- métricas ---------- */
  div[data-testid="stMetric"] {
      background:#fff; border:1px solid #E3ECE6; border-left:5px solid var(--verde);
      border-radius:14px; padding:.85rem 1rem; box-shadow:0 4px 14px rgba(11,37,69,.05);
  }
  div[data-testid="stMetric"] label { color:#5A6B7B !important; font-size:.78rem; }
  div[data-testid="stMetricValue"] { color:var(--azul); font-weight:800; font-size:1.3rem; }

  /* ---------- contenedores / pestañas / botones ---------- */
  div[data-testid="stVerticalBlockBorderWrapper"] {
      border-radius:16px; border-color:#E3ECE6 !important; background:#fff;
  }
  .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
  .stTabs [data-baseweb="tab"] {
      border-radius:10px 10px 0 0; padding:.45rem .9rem; font-weight:600; color:#5A6B7B;
  }
  .stTabs [aria-selected="true"] { color:var(--azul); background:#fff; }
  .stButton > button, .stDownloadButton > button {
      border-radius:10px; font-weight:600; border:1px solid #D7E4DC;
  }
  .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
      background:linear-gradient(135deg,#1E9E5A,#2EC27E); border:0; color:#fff;
      box-shadow:0 6px 16px rgba(30,158,90,.28);
  }
  .stButton > button[kind="primary"]:hover { filter:brightness(1.05); }
  div[data-testid="stDataFrame"] { border:1px solid #E3ECE6; border-radius:12px; overflow:hidden; }
  .nota {
      background:#FFF8EC; border:1px solid #F4D9A6; border-left:5px solid #F4A261;
      border-radius:12px; padding:.75rem 1rem; color:#6B4E1F; font-size:.88rem;
  }
  .ok {
      background:#EAF8F0; border:1px solid #BFE8CE; border-left:5px solid #1E9E5A;
      border-radius:12px; padding:.75rem 1rem; color:#14532D; font-size:.88rem;
  }
  .step { background:#fff; border:1px solid #E3ECE6; border-radius:14px; padding:1rem 1.1rem; height:100%; }
  .step .n {
      width:30px;height:30px;border-radius:9px;display:grid;place-items:center;
      background:#E4F4EA;color:#1E9E5A;font-weight:800;margin-bottom:.55rem;
  }
  .step b { color:var(--azul); }
  .step p { color:#5A6B7B; font-size:.85rem; margin:.25rem 0 0; }
  .footer { text-align:center; color:#8A99A8; font-size:.76rem; padding:2rem 0 .5rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Utilidades de persistencia
# --------------------------------------------------------------------------- #
def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return name[:80] or "archivo"


def save_dataset(df: pd.DataFrame, costs: pd.DataFrame | None, source: str) -> None:
    """Guarda el dataset limpio y el catálogo de costos para sobrevivir reinicios."""
    try:
        CURRENT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CURRENT_DIR / "dataset.csv", index=False)
        costs_path = CURRENT_DIR / "costs.csv"
        if costs is not None and len(costs):
            costs.to_csv(costs_path, index=False)
        elif costs_path.exists():
            costs_path.unlink()
        (CURRENT_DIR / "meta.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "rows": int(len(df)),
                    "columns": int(df.shape[1]),
                    "log": list(S.get("log", [])),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # la persistencia nunca debe romper la app
        st.warning(f"No se pudo guardar la sesión en disco: {exc}")


def load_saved() -> tuple[pd.DataFrame, pd.DataFrame | None, dict] | None:
    """Recupera la última sesión guardada (datos + costos + metadatos)."""
    f = CURRENT_DIR / "dataset.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
        for c in dp.DATES:
            if c in df:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        costs = None
        cf = CURRENT_DIR / "costs.csv"
        if cf.exists():
            costs = pd.read_csv(cf)
        meta = {}
        mf = CURRENT_DIR / "meta.json"
        if mf.exists():
            meta = json.loads(mf.read_text(encoding="utf-8"))
        return df, costs, meta
    except Exception:
        return None


def clear_saved() -> None:
    for name in ("dataset.csv", "costs.csv", "meta.json"):
        p = CURRENT_DIR / name
        if p.exists():
            p.unlink()


def model_on_disk(kind: str) -> Path:
    return MODELS_DIR / MODEL_FILES[kind]


def load_model_from_disk(kind: str) -> dict | None:
    """Carga un modelo entrenado previamente. Devuelve un resumen para la UI."""
    p = model_on_disk(kind)
    if not p.exists():
        return None
    try:
        b = joblib.load(p)
        out = {
            "viable": True,
            "from_disk": True,
            "mejor": b.get("nombre", "modelo guardado"),
            "metrics": b.get("metrics"),
            "umbrales": b.get("umbrales"),
            "tipo_resultado": b.get("tipo_resultado"),
        }
        return out
    except Exception:
        return None


def hydrate_once() -> None:
    """Al abrir la app: recupera datos y modelos persistidos (solo una vez)."""
    if S.get("_booted"):
        return
    S["_booted"] = True
    saved = load_saved()
    if saved:
        df, costs, meta = saved
        S["df"], S["costs"], S["meta"] = df, costs, meta
        S["source"] = meta.get("source", "sesión guardada")
        S["log"] = meta.get("log") or ["Sesión recuperada desde el almacenamiento persistente."]
    elif os.getenv("AUTO_LOAD_DEMO", "0") == "1" and DEMO_FILE.exists():
        try:
            raw = dp.load_file(DEMO_FILE)
            df, log = dp.clean(raw)
            S["df"], S["log"], S["source"] = df, log, DEMO_FILE.name
            S["demo"] = True
        except Exception:
            pass
    for kind in ("prof", "ret"):
        if kind not in S:
            m = load_model_from_disk(kind)
            if m:
                S[kind] = m


hydrate_once()


# --------------------------------------------------------------------------- #
# Utilidades de UI
# --------------------------------------------------------------------------- #
def page_head(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-head"><div class="icon">{icon}</div>'
        f"<div><h1>{title}</h1><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="kicker">Mercado Libre · Analítica con IA</div>
  <h1>Rentabilidad y riesgo de devoluciones,<br>antes de que sea un problema</h1>
  <p>Sube tu reporte de ventas y obtén indicadores económicos, predicciones con
     validación temporal, factores asociados y recomendaciones accionables.
     Todo el procesamiento ocurre de forma local y sin datos personales.</p>
  <span class="pill">🔒 Sin PII</span>
  <span class="pill">📈 Validación temporal</span>
  <span class="pill">🤖 Modelos comparados</span>
  <span class="pill">💾 Persistencia de datos y modelos</span>
</div>
""",
        unsafe_allow_html=True,
    )


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return "N/D" if pd.isna(x) else f"{x:.1%}"


def chart(fig: go.Figure, height: int | None = None) -> None:
    fig.update_layout(
        template="plotly_white",
        colorway=COLORES,
        margin=dict(l=10, r=10, t=52, b=64),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=AZUL),
        title=dict(font=dict(size=15, color=AZUL), x=0.01),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def data_table(df: pd.DataFrame, height: int | None = None) -> None:
    kwargs = {"height": height} if height else {}
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def go_to(section: str) -> None:
    S["menu"] = section


def require_data() -> pd.DataFrame:
    if "df" not in S:
        st.markdown(
            '<div class="nota">Primero carga tu reporte en <b>Carga de datos</b> '
            "o usa el conjunto de demostración.</div>",
            unsafe_allow_html=True,
        )
        st.stop()
    return S["df"]


def filtered() -> pd.DataFrame:
    """Datos con economía calculada + filtros globales de la barra lateral."""
    d = pm.add_economics(require_data(), S.get("costs"))
    with st.sidebar:
        st.markdown("#### Filtros")
        if "fecha_venta" in d and d.fecha_venta.notna().any():
            lo = d.fecha_venta.min().date()
            hi = d.fecha_venta.max().date()
            rng = st.date_input("Rango de fechas", (lo, hi), min_value=lo, max_value=hi)
            if isinstance(rng, (tuple, list)) and rng:
                a = rng[0]
                b = rng[1] if len(rng) > 1 else rng[0]
                d = d[(d.fecha_venta.dt.date >= a) & (d.fecha_venta.dt.date <= b)]
        if "sku" in d:
            skus = sorted(d.sku.dropna().astype(str).unique())
            picked = st.multiselect("SKU", skus)
            if picked:
                d = d[d.sku.astype(str).isin(picked)]
        st.caption(f"{len(d):,} de {len(S['df']):,} ventas en el filtro")
    return d


def load_demo() -> bool:
    """Carga el CSV simulado incluido en el proyecto."""
    if not DEMO_FILE.exists():
        st.error("No se encontró `data/demo_simulado.csv`.")
        return False
    raw = dp.load_file(DEMO_FILE)
    df, log = dp.clean(raw)
    S["df"], S["log"], S["source"], S["demo"] = df, log, DEMO_FILE.name, True
    S.pop("prof", None)
    S.pop("ret", None)
    save_dataset(df, S.get("costs"), DEMO_FILE.name)
    return True


def train_profitability() -> None:
    with st.spinner("Entrenando y comparando modelos de rentabilidad…"):
        r = pm.train(S["df"], S.get("costs"), models_dir=str(MODELS_DIR))
    S["prof"] = r
    if r.get("viable"):
        st.toast(f"Modelo de rentabilidad listo: {r['mejor']}", icon="✅")
    else:
        st.toast("El modelo de rentabilidad no es viable con estos datos", icon="⚠️")


def train_returns() -> None:
    with st.spinner("Entrenando y comparando modelos de devoluciones…"):
        r = rm.train(S["df"], models_dir=str(MODELS_DIR))
    S["ret"] = r
    if r.get("viable"):
        st.toast(f"Modelo de devoluciones listo: {r['mejor']}", icon="✅")
    else:
        st.toast("El modelo de devoluciones no es viable con estos datos", icon="⚠️")


def model_badge(kind: str) -> str:
    m = S.get(kind)
    if m and m.get("viable"):
        origen = "guardado" if m.get("from_disk") else "en sesión"
        return f'<span class="dot on"></span>{m.get("mejor", "modelo")} · {origen}'
    return '<span class="dot off"></span>sin entrenar'


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">📦</div>'
        "<div><b>ML Analytics</b><span>Mercado Libre · IA</span></div></div>",
        unsafe_allow_html=True,
    )
    sec = st.radio("Menú", SECCIONES, key="menu", format_func=lambda s: f"{ICONOS[s]}  {s}")
    st.divider()

    if "df" in S:
        df0 = S["df"]
        rango = ""
        if "fecha_venta" in df0 and df0.fecha_venta.notna().any():
            rango = (
                f"{df0.fecha_venta.min():%d/%m/%Y} – {df0.fecha_venta.max():%d/%m/%Y}"
            )
        n_skus = df0.sku.nunique() if "sku" in df0 else "N/D"
        st.markdown(
            '<div class="side-card"><b>Conjunto activo</b><br>'
            f"📄 {S.get('source', 'datos')}<br>"
            f"🧾 {len(df0):,} filas · {n_skus} SKUs<br>"
            f"🗓️ {rango or 'sin fechas válidas'}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="side-card"><b>Sin datos cargados</b><br>'
            "Sube tu reporte o usa el conjunto de demostración.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="side-card"><b>Modelos</b><br>'
        f"{model_badge('prof')}<br>{model_badge('ret')}</div>",
        unsafe_allow_html=True,
    )
    st.caption("El archivo original nunca se modifica. Datos personales excluidos.")


# --------------------------------------------------------------------------- #
# Inicio
# --------------------------------------------------------------------------- #
if sec == "Inicio":
    hero()

    if "df" not in S:
        st.markdown("### Empieza en tres pasos")
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            '<div class="step"><div class="n">1</div><b>Carga tu reporte</b>'
            "<p>Excel o CSV exportado de Mercado Libre. Detectamos encabezados y limpiamos importes y fechas.</p></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            '<div class="step"><div class="n">2</div><b>Analiza y entrena</b>'
            "<p>Indicadores económicos, riesgo de devoluciones y comparación de modelos con validación temporal.</p></div>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            '<div class="step"><div class="n">3</div><b>Decide</b>'
            "<p>Recomendaciones por SKU, transportista y envío. Exporta reportes en CSV.</p></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        a, b, _ = st.columns([1.1, 1.4, 2])
        if a.button("📤 Ir a Carga de datos", type="primary", width="stretch"):
            S["menu"] = "Carga de datos"
            st.rerun()
        if b.button("🚀 Explorar con datos de demostración", width="stretch"):
            with st.spinner("Cargando datos simulados…"):
                if load_demo():
                    st.rerun()
        st.markdown(
            '<div class="nota">Los datos de demostración son <b>simulados</b>: sirven para probar '
            "el sistema, pero sus métricas no representan resultados reales.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    d = filtered()
    if not len(d):
        st.warning("El filtro actual no devuelve ventas. Amplía el rango de fechas o los SKUs.")
        st.stop()

    parcial = bool((d.tipo_resultado != "estimado con costo").any())
    dg = rm.diagnose(d)
    ver = dg["devoluciones(1)"] + dg["no_devoluciones(0)"]
    tasa_dev = dg["devoluciones(1)"] / ver if ver else np.nan

    c = st.columns(5)
    c[0].metric("Ventas", f"{len(d):,}", help="Filas del reporte dentro del filtro actual.")
    c[1].metric(
        "Ingresos por producto",
        money(d.get("ingresos_productos", pd.Series(dtype=float)).sum()),
        help="Ingresos brutos por productos antes de cargos, envíos y reembolsos.",
    )
    c[2].metric(
        ("Resultado PARCIAL" if parcial else "Resultado estimado"),
        money(d.resultado_economico.sum()),
        help="Total de Mercado Libre menos costo de adquisición (si existe catálogo).",
    )
    c[3].metric("SKUs activos", d.sku.nunique() if "sku" in d else "N/D")
    c[4].metric(
        "Tasa de devolución",
        pct(tasa_dev),
        help=f"{dg['devoluciones(1)']} devoluciones verificadas sobre {ver} ventas con resultado verificable.",
    )

    if parcial:
        st.markdown(
            '<div class="nota">Sin catálogo de costos el resultado es <b>parcial</b> '
            "(no es utilidad neta). Cárgalo en <i>Carga de datos</i> para obtener el resultado estimado.</div>",
            unsafe_allow_html=True,
        )

    t1, t2, t3 = st.tabs(["📈 Evolución mensual", "🏆 Top SKUs", "🧾 Actividad reciente"])

    with t1:
        if "fecha_venta" in d and d.fecha_venta.notna().any():
            m = (
                d.set_index("fecha_venta")
                .resample("MS")
                .agg(ventas=("id_venta", "count"), resultado=("resultado_economico", "sum"))
                .reset_index()
            )
            fig = go.Figure()
            fig.add_bar(
                x=m.fecha_venta, y=m.ventas, name="Ventas", marker_color=VERDE,
                hovertemplate="%{x|%b %Y}<br>%{y} ventas<extra></extra>",
            )
            fig.add_scatter(
                x=m.fecha_venta, y=m.resultado, name="Resultado (MXN)", yaxis="y2",
                mode="lines+markers", line=dict(color=AZUL_2, width=3),
                hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
            )
            fig.update_layout(
                title="Ventas y resultado económico por mes",
                yaxis=dict(title="Ventas"),
                yaxis2=dict(title="Resultado (MXN)", overlaying="y", side="right", showgrid=False),
                hovermode="x unified",
            )
            chart(fig, 380)
        else:
            st.info("El conjunto no tiene fechas de venta válidas.")

    with t2:
        if "sku" in d:
            g = (
                d.groupby("sku")
                .agg(
                    ventas=("id_venta", "count"),
                    ingresos=("ingresos_productos", "sum"),
                    resultado=("resultado_economico", "sum"),
                )
                .reset_index()
            )
            a, b = st.columns(2)
            with a:
                chart(
                    px.bar(
                        g.nlargest(10, "ventas").sort_values("ventas"),
                        x="ventas", y="sku", orientation="h",
                        title="SKUs con más ventas", color_discrete_sequence=[VERDE],
                    ),
                    360,
                )
            with b:
                chart(
                    px.bar(
                        g.nlargest(10, "resultado").sort_values("resultado"),
                        x="resultado", y="sku", orientation="h",
                        title="SKUs con mayor resultado", color_discrete_sequence=[AZUL_2],
                    ),
                    360,
                )
        else:
            st.info("El conjunto no incluye la columna SKU.")

    with t3:
        cols = [
            c
            for c in ["id_venta", "fecha_venta", "sku", "estado", "unidades",
                      "ingresos_productos", "resultado_economico", "tipo_resultado"]
            if c in d
        ]
        data_table(d[cols].sort_values("fecha_venta", ascending=False).head(25), 380)


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #
elif sec == "Carga de datos":
    page_head("📤", "Carga de datos", "Sube tu reporte de Mercado Libre y, si lo tienes, tu catálogo de costos.")

    izq, der = st.columns([1.25, 1])

    with izq:
        st.markdown("#### 1 · Reporte de ventas")
        f = st.file_uploader(
            "Archivo .xlsx, .xls o .csv exportado de Mercado Libre",
            type=["xlsx", "xls", "csv"],
            label_visibility="collapsed",
        )
        if f is not None:
            firma = (f.name, f.size)
            if S.get("_upload_sig") != firma:
                try:
                    with st.spinner("Leyendo y limpiando el archivo…"):
                        raw = dp.load_file(f)
                        df, log = dp.clean(raw)
                    S["df"], S["log"], S["source"] = df, log, f.name
                    S.pop("demo", None)
                    S.pop("prof", None)
                    S.pop("ret", None)
                    S["_upload_sig"] = firma
                    save_dataset(df, S.get("costs"), f.name)
                    st.toast(f"{len(df):,} filas cargadas y guardadas", icon="✅")
                    st.rerun()
                except Exception as exc:
                    st.error(f"No se pudo leer el archivo: {exc}")
                    S.pop("_upload_sig", None)

        if st.button("🚀 Usar datos de demostración", width="stretch"):
            with st.spinner("Cargando datos simulados…"):
                if load_demo():
                    st.rerun()

    with der:
        st.markdown("#### 2 · Catálogo de costos (opcional)")
        cf = st.file_uploader(
            "CSV o Excel con columnas sku, costo_unitario",
            type=["csv", "xlsx"],
            key="costs_upload",
            label_visibility="collapsed",
        )
        if cf is not None:
            firma = (cf.name, cf.size)
            if S.get("_costs_sig") != firma:
                try:
                    costs = pd.read_csv(cf) if cf.name.lower().endswith(".csv") else pd.read_excel(cf)
                    costs.columns = [str(c).strip().lower() for c in costs.columns]
                    if not {"sku", "costo_unitario"} <= set(costs.columns):
                        raise ValueError("El catálogo debe tener las columnas 'sku' y 'costo_unitario'.")
                    S["costs"] = costs
                    S["_costs_sig"] = firma
                    if "df" in S:
                        save_dataset(S["df"], costs, S.get("source", "datos"))
                    st.toast(f"{len(costs):,} costos cargados", icon="✅")
                    st.rerun()
                except Exception as exc:
                    st.error(f"No se pudo leer el catálogo: {exc}")
        if S.get("costs") is not None:
            st.success(f"Catálogo activo: {len(S['costs']):,} SKUs con costo unitario.")
            st.caption("El resultado económico se recalcula automáticamente con estos costos.")
        else:
            st.info("Sin costos, el resultado se marca como **PARCIAL**.")

    st.divider()

    if "df" not in S:
        st.info("Aún no hay datos cargados. Sube un archivo o usa el conjunto de demostración.")
        st.stop()

    df = S["df"]
    rep = dp.quality_report(df)
    meta = S.get("meta") or {}
    guardado = f" · guardado {meta.get('saved_at')}" if meta.get("saved_at") else ""
    st.markdown(
        f'<div class="ok"><b>{len(df):,} filas · {df.shape[1]} columnas</b> · '
        f"fuente: {S.get('source', 'datos')}{guardado}. Datos personales excluidos.</div>",
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(
        ["🧹 Limpieza", "📋 Calidad", "🔄 Devoluciones", "👀 Vista previa"]
    )
    with t1:
        st.markdown("**Decisiones aplicadas**")
        for x in S.get("log", []):
            st.markdown(f"- {x}")
    with t2:
        a, b, c = st.columns(3)
        a.metric("Filas", f"{rep['filas']:,}")
        b.metric("Columnas", rep["columnas"])
        c.metric("IDs de venta repetidos", rep.get("id_venta_repetidos", "N/D"))
        x, y = st.columns(2)
        with x:
            st.write("**Inconsistencias**")
            inc = rep.get("inconsistencias", {})
            data_table(pd.DataFrame({"hallazgo": list(inc), "casos": list(inc.values())}) if inc else pd.DataFrame({"hallazgo": ["Sin hallazgos"]}))
        with y:
            st.write("**Atípicos (IQR × 3)**")
            at = rep.get("atipicos_iqr3", {})
            data_table(pd.DataFrame({"columna": list(at), "casos": list(at.values())}) if at else pd.DataFrame({"columna": ["Sin atípicos"]}))
        st.write("**Porcentaje de valores faltantes**")
        falt = rep["faltantes_pct"].reset_index()
        falt.columns = ["columna", "faltantes_pct"]
        data_table(falt.head(20), 300)
    with t3:
        dg = rm.diagnose(df)
        a, b, c, dcol = st.columns(4)
        a.metric("Devoluciones (1)", f"{dg['devoluciones(1)']:,}")
        b.metric("No devoluciones (0)", f"{dg['no_devoluciones(0)']:,}")
        c.metric("Ambiguas (excluidas)", f"{dg['ambiguas(excluidas)']:,}")
        ver = dg["devoluciones(1)"] + dg["no_devoluciones(0)"]
        dcol.metric("Tasa verificable", pct(dg["devoluciones(1)"] / ver) if ver else "N/D")
        st.caption(
            "Solo se etiqueta 1 cuando hay evidencia de devolución y 0 cuando la entrega "
            "tiene ≥30 días sin reclamo ni devolución. El resto se excluye del entrenamiento."
        )
    with t4:
        data_table(df.head(100), 420)
        st.download_button(
            "⬇️ Descargar datos procesados (CSV)",
            df.to_csv(index=False).encode("utf-8-sig"),
            "datos_procesados.csv",
            mime="text/csv",
        )

    with st.expander("🗑️ Eliminar datos guardados"):
        st.write("Borra el conjunto persistido en disco y reinicia la sesión de datos.")
        if st.button("Eliminar datos y empezar de nuevo", type="secondary"):
            clear_saved()
            for k in ("df", "costs", "log", "meta", "source", "prof", "ret", "_upload_sig", "_costs_sig", "demo"):
                S.pop(k, None)
            st.rerun()


# --------------------------------------------------------------------------- #
# Análisis exploratorio
# --------------------------------------------------------------------------- #
elif sec == "Análisis exploratorio":
    page_head("📊", "Análisis exploratorio", "Panorama de productos, envíos y evolución del negocio.")
    d = filtered()
    if not len(d):
        st.warning("El filtro actual no devuelve ventas.")
        st.stop()

    t1, t2, t3 = st.tabs(["🛍️ Productos", "🚚 Envíos", "💵 Importes"])

    with t1:
        if "sku" in d:
            g = (
                d.groupby("sku")
                .agg(
                    ventas=("id_venta", "count"),
                    ingresos=("ingresos_productos", "sum"),
                    resultado=("resultado_economico", "sum"),
                )
                .reset_index()
                .sort_values("ventas", ascending=False)
            )
            a, b = st.columns(2)
            with a:
                chart(
                    px.bar(
                        g.nlargest(10, "ventas").sort_values("ventas"),
                        x="ventas", y="sku", orientation="h",
                        title="Más vendidos (unidades de venta)",
                        color_discrete_sequence=[VERDE],
                    ),
                    380,
                )
            with b:
                chart(
                    px.bar(
                        g.nlargest(10, "ingresos").sort_values("ingresos"),
                        x="ingresos", y="sku", orientation="h",
                        title="Mayores ingresos por producto",
                        color_discrete_sequence=[AZUL_2],
                    ),
                    380,
                )
            st.write("**Detalle por SKU**")
            data_table(g, 320)
        else:
            st.info("El conjunto no incluye SKU.")

    with t2:
        if "transportista" in d and "costos_envio" in d:
            t = (
                d.groupby("transportista")
                .agg(
                    ventas=("id_venta", "count"),
                    costo_envio=("costos_envio", lambda s: s.abs().mean()),
                    costo_total=("costos_envio", lambda s: s.abs().sum()),
                )
                .reset_index()
                .sort_values("costo_envio", ascending=False)
            )
            a, b = st.columns(2)
            with a:
                chart(
                    px.bar(
                        t, x="transportista", y="costo_envio",
                        title="Costo de envío promedio por transportista",
                        color="costo_envio", color_continuous_scale=["#A8DADC", VERDE],
                    ),
                    380,
                )
            with b:
                chart(
                    px.scatter(
                        t, x="ventas", y="costo_envio", size="costo_total",
                        color="transportista", hover_name="transportista",
                        title="Volumen vs costo promedio de envío",
                    ),
                    380,
                )
            data_table(t, 260)
        else:
            st.info("Se requieren las columnas `transportista` y `costos_envio`.")
        a, b = st.columns(2)
        if "descuentos" in d:
            a.metric("Descuentos y bonificaciones", money(d["descuentos"].sum()))
        if "anulaciones" in d:
            b.metric("Anulaciones y reembolsos", money(d["anulaciones"].sum()))

    with t3:
        if "precio_unitario" in d:
            chart(
                px.histogram(
                    d.dropna(subset=["precio_unitario"]),
                    x="precio_unitario", nbins=40,
                    title="Distribución del precio unitario",
                    color_discrete_sequence=[VERDE],
                ),
                360,
            )
        if "resultado_economico" in d:
            chart(
                px.box(
                    d, x="tipo_resultado", y="resultado_economico",
                    title="Resultado económico por tipo de cálculo",
                    color="tipo_resultado", color_discrete_sequence=[VERDE, AMBAR],
                ),
                360,
            )
        cols = [c for c in ["ingresos_productos", "cargo_venta", "ingresos_envio",
                            "costos_envio", "descuentos", "anulaciones", "resultado_economico"] if c in d]
        if cols:
            st.write("**Resumen estadístico (MXN)**")
            data_table(d[cols].describe().T.round(2), 300)


# --------------------------------------------------------------------------- #
# Rentabilidad
# --------------------------------------------------------------------------- #
elif sec == "Rentabilidad":
    page_head("💰", "Rentabilidad", "Resultado económico por venta y modelos de regresión comparados.")
    d = filtered()

    izq, der = st.columns([1.2, 2])
    with izq:
        st.markdown("#### Entrenamiento")
        st.caption(
            "Se comparan baseline, regresión lineal, Random Forest y XGBoost. "
            "El 80 % más antiguo entrena y el 20 % más reciente prueba (validación temporal)."
        )
        if st.button("🤖 Entrenar / reentrenar", type="primary", width="stretch"):
            train_profitability()
            st.rerun()
        if S.get("prof", {}).get("from_disk"):
            st.info("Modelo recuperado del disco. Reentrena para ver importancia y predicciones.")
    with der:
        r = S.get("prof")
        if r and r.get("viable"):
            m = r["metrics"]
            mejor_row = m[m["Modelo"] == r["mejor"]].iloc[0]
            a, b, c = st.columns(3)
            a.metric("Mejor modelo", r["mejor"])
            b.metric("RMSE (prueba)", f"{mejor_row['RMSE']:,.2f}")
            c.metric("R² (prueba)", f"{mejor_row['R2']:.3f}")
            if r.get("periodo_train"):
                st.caption(
                    f"Entrenamiento: {r['periodo_train'][0]:%d/%m/%Y} → {r['periodo_train'][1]:%d/%m/%Y} · "
                    f"Prueba: {r['periodo_test'][0]:%d/%m/%Y} → {r['periodo_test'][1]:%d/%m/%Y}"
                )
            data_table(m.round(3).rename(columns={"R2": "R²"}))
        elif r and not r.get("viable"):
            st.error(r.get("razon", "Modelo no viable."))
        else:
            st.markdown(
                '<div class="nota">Aún no hay modelo de rentabilidad. Pulsa '
                "<b>Entrenar / reentrenar</b> para comparar algoritmos.</div>",
                unsafe_allow_html=True,
            )

    r = S.get("prof")
    if r and r.get("viable"):
        if r.get("importancia") is not None:
            st.divider()
            a, b = st.columns([1.15, 1])
            with a:
                imp = r["importancia"].head(10).sort_values("importancia")
                chart(
                    px.bar(
                        imp, x="importancia", y="variable", orientation="h",
                        title="Importancia por permutación (asociación, no causalidad)",
                        color="importancia", color_continuous_scale=["#A8DADC", VERDE],
                    ),
                    400,
                )
            with b:
                if r.get("predicciones") is not None and len(r["predicciones"]):
                    p = r["predicciones"]
                    lo = float(min(p.real.min(), p.predicho.min()))
                    hi = float(max(p.real.max(), p.predicho.max()))
                    fig = go.Figure()
                    fig.add_scatter(
                        x=p.real, y=p.predicho, mode="markers", name="Ventas",
                        marker=dict(color=VERDE, size=7, opacity=.6),
                    )
                    fig.add_scatter(
                        x=[lo, hi], y=[lo, hi], mode="lines", name="Predicción perfecta",
                        line=dict(color=AZUL_2, dash="dash"),
                    )
                    fig.update_layout(title="Real vs predicho (conjunto de prueba)",
                                      xaxis_title="Resultado real (MXN)",
                                      yaxis_title="Resultado predicho (MXN)")
                    chart(fig, 400)
                else:
                    st.info("Reentrena el modelo para generar las predicciones de prueba.")

    if "sku" in d and len(d):
        st.divider()
        st.markdown("#### Resultado económico por SKU")
        g = d.groupby("sku").resultado_economico.sum().sort_values()
        if len(g) <= 20:
            st.write("**Ranking completo**")
            data_table(
                g.reset_index().rename(columns={"resultado_economico": "resultado"}),
                360,
            )
        else:
            a, b = st.columns(2)
            with a:
                st.write("**Menor resultado**")
                data_table(g.head(10).reset_index().rename(columns={"resultado_economico": "resultado"}), 320)
            with b:
                st.write("**Mejor resultado**")
                data_table(g.tail(10)[::-1].reset_index().rename(columns={"resultado_economico": "resultado"}), 320)


# --------------------------------------------------------------------------- #
# Predicción de devoluciones
# --------------------------------------------------------------------------- #
elif sec == "Predicción de devoluciones":
    page_head("🔄", "Predicción de devoluciones", "Clasificación del riesgo relativo con evidencia verificable.")
    d = require_data()
    dg = rm.diagnose(d)
    ver = dg["devoluciones(1)"] + dg["no_devoluciones(0)"]
    a, b, c, e = st.columns(4)
    a.metric("Devoluciones (1)", f"{dg['devoluciones(1)']:,}")
    b.metric("No devoluciones (0)", f"{dg['no_devoluciones(0)']:,}")
    c.metric("Ambiguas (excluidas)", f"{dg['ambiguas(excluidas)']:,}")
    e.metric("Tasa verificable", pct(dg["devoluciones(1)"] / ver) if ver else "N/D")

    izq, der = st.columns([1.2, 2])
    with izq:
        st.markdown("#### Entrenamiento")
        st.caption(
            "Se comparan baseline, regresión logística, árbol de decisión, Random Forest y XGBoost. "
            "El riesgo es relativo: las probabilidades no están calibradas."
        )
        if st.button("🤖 Entrenar / reentrenar", type="primary", width="stretch"):
            train_returns()
            st.rerun()
        if S.get("ret", {}).get("from_disk"):
            st.info("Modelo recuperado del disco. Reentrena para ver importancia y umbrales nuevos.")
    with der:
        r = S.get("ret")
        if r and r.get("viable"):
            a, b, c = st.columns(3)
            a.metric("Mejor modelo", r["mejor"])
            if r.get("metrics") is not None:
                best = r["metrics"][r["metrics"]["Modelo"] == r["mejor"]].iloc[0]
                b.metric("ROC-AUC", f"{best['ROC-AUC']:.3f}")
                c.metric("F1", f"{best['F1']:.3f}")
                data_table(r["metrics"].round(3))
        elif r and not r.get("viable"):
            st.error(r.get("razon", "Modelo no viable."))
        else:
            st.markdown(
                '<div class="nota">Aún no hay modelo de devoluciones. Pulsa '
                "<b>Entrenar / reentrenar</b>.</div>",
                unsafe_allow_html=True,
            )

    r = S.get("ret")
    if r and r.get("viable"):
        if r.get("importancia") is not None:
            st.divider()
            chart(
                px.bar(
                    r["importancia"].head(10).sort_values("importancia"),
                    x="importancia", y="variable", orientation="h",
                    title="Factores asociados al riesgo de devolución",
                    color="importancia", color_continuous_scale=["#F6D5A8", ROJO],
                ),
                400,
            )

        if model_on_disk("ret").exists():
            st.divider()
            st.markdown("#### Riesgo relativo por venta")
            with st.spinner("Calculando probabilidades…"):
                p = rm.predict(d, models_dir=str(MODELS_DIR))
            out = d.join(p)
            counts = out.riesgo.value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0)
            colores = {"Alto": ROJO, "Medio": AMBAR, "Bajo": VERDE}
            chart(
                px.bar(
                    x=counts.index, y=counts.values, color=counts.index,
                    color_discrete_map=colores,
                    title="Ventas por banda de riesgo",
                    labels={"x": "Riesgo", "y": "Ventas", "color": "Riesgo"},
                ),
                320,
            )
            cols = [c for c in ["id_venta", "sku", "transportista", "prob_devolucion", "riesgo"] if c in out]
            tabla = out[cols].sort_values("prob_devolucion", ascending=False).head(100)
            if "prob_devolucion" in tabla:
                tabla["prob_devolucion"] = tabla["prob_devolucion"].round(3)
            data_table(tabla, 420)
            st.caption("Bajo / Medio / Alto se definen por percentiles de la probabilidad estimada.")
            st.download_button(
                "⬇️ Descargar riesgo por venta (CSV)",
                out[cols].to_csv(index=False).encode("utf-8-sig"),
                "riesgo_devoluciones.csv",
                mime="text/csv",
            )


# --------------------------------------------------------------------------- #
# Comparación de modelos
# --------------------------------------------------------------------------- #
elif sec == "Comparación de modelos":
    page_head("⚖️", "Comparación de modelos", "Métricas lado a lado para elegir con criterio.")
    t1, t2 = st.tabs(["💰 Rentabilidad", "🔄 Devoluciones"])

    with t1:
        r = S.get("prof")
        if r and r.get("viable") and r.get("metrics") is not None:
            m = r["metrics"]
            st.success(f"Mejor modelo (menor RMSE): **{r['mejor']}**")
            data_table(m.round(3).rename(columns={"R2": "R²"}))
            chart(
                px.bar(
                    m.sort_values("RMSE"), x="Modelo", y="RMSE",
                    title="RMSE por modelo (menor es mejor)",
                    color="Modelo", color_discrete_sequence=COLORES,
                ),
                340,
            )
            st.caption("RMSE y MAE en MXN. R² compara contra predecir la media.")
        else:
            st.info("Entrena el modelo de rentabilidad para ver la comparación.")

    with t2:
        r = S.get("ret")
        if r and r.get("viable") and r.get("metrics") is not None:
            m = r["metrics"]
            st.success(f"Mejor modelo (mayor ROC-AUC): **{r['mejor']}**")
            data_table(m.round(3))
            chart(
                px.bar(
                    m.sort_values("ROC-AUC"), x="Modelo", y="ROC-AUC",
                    title="ROC-AUC por modelo (mayor es mejor)",
                    color="Modelo", color_discrete_sequence=COLORES,
                ),
                340,
            )
            st.caption(
                "La matriz de confusión se calcula con umbral 0.5. "
                "Con clases desbalanceadas, ROC-AUC y F1 son más informativos que Accuracy."
            )
        else:
            st.info("Entrena el modelo de devoluciones para ver la comparación.")


# --------------------------------------------------------------------------- #
# Recomendaciones
# --------------------------------------------------------------------------- #
elif sec == "Recomendaciones inteligentes":
    page_head("💡", "Recomendaciones inteligentes", "Hallazgos accionables basados en los datos observados.")
    d = filtered()
    if not len(d):
        st.warning("El filtro actual no devuelve ventas.")
        st.stop()

    out = rec.recommend(d)
    if not len(out):
        st.markdown(
            '<div class="nota">No hay hallazgos con el mínimo de ventas por SKU. '
            "Prueba con más historial o amplía los filtros.</div>",
            unsafe_allow_html=True,
        )
    else:
        counts = out.tipo.value_counts()
        cols = st.columns(min(len(counts), 4))
        for col, (tipo, n) in zip(cols, counts.items()):
            col.metric(tipo, int(n))
        for tipo in counts.index:
            grupo = out[out.tipo == tipo]
            with st.expander(f"{tipo} · {len(grupo)} hallazgo(s)", expanded=True):
                data_table(grupo[["elemento", "evidencia"]], 260)
        st.download_button(
            "⬇️ Descargar recomendaciones (CSV)",
            out.to_csv(index=False).encode("utf-8-sig"),
            "recomendaciones.csv",
            mime="text/csv",
        )
    st.caption("Las recomendaciones describen asociaciones observadas; no demuestran causalidad.")


# --------------------------------------------------------------------------- #
# Reportes
# --------------------------------------------------------------------------- #
elif sec == "Reportes":
    page_head("📄", "Reportes", "Exporta los resultados de la sesión actual.")
    d = filtered()

    st.markdown("#### Descargas")
    a, b, c = st.columns(3)
    with a:
        st.download_button(
            "⬇️ Datos con resultado económico",
            d.to_csv(index=False).encode("utf-8-sig"),
            "reporte_economico.csv",
            mime="text/csv",
            width="stretch",
        )
    with b:
        prof = S.get("prof")
        if prof and prof.get("viable") and prof.get("metrics") is not None:
            st.download_button(
                "⬇️ Métricas de rentabilidad",
                prof["metrics"].to_csv(index=False).encode("utf-8-sig"),
                "metricas_rentabilidad.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("Métricas de rentabilidad", disabled=True, width="stretch")
    with c:
        ret = S.get("ret")
        if ret and ret.get("viable") and ret.get("metrics") is not None:
            st.download_button(
                "⬇️ Métricas de devoluciones",
                ret["metrics"].to_csv(index=False).encode("utf-8-sig"),
                "metricas_devoluciones.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("Métricas de devoluciones", disabled=True, width="stretch")

    st.divider()
    st.markdown("#### Resumen de la sesión")
    parcial = bool((d.tipo_resultado != "estimado con costo").any())
    dg = rm.diagnose(d)
    ver = dg["devoluciones(1)"] + dg["no_devoluciones(0)"]
    resumen = pd.DataFrame(
        {
            "Indicador": [
                "Fuente de datos", "Ventas en el filtro", "SKUs", "Ingresos por producto",
                "Resultado económico", "Tipo de resultado", "Devoluciones verificadas",
                "Tasa de devolución verificable", "Modelo de rentabilidad", "Modelo de devoluciones",
            ],
            "Valor": [
                S.get("source", "datos"),
                f"{len(d):,}",
                f"{d.sku.nunique() if 'sku' in d else 'N/D'}",
                money(d.get("ingresos_productos", pd.Series(dtype=float)).sum()),
                money(d.resultado_economico.sum()),
                "PARCIAL (sin costos)" if parcial else "Estimado con costo",
                f"{dg['devoluciones(1)']:,}",
                pct(dg["devoluciones(1)"] / ver) if ver else "N/D",
                S.get("prof", {}).get("mejor", "No entrenado"),
                S.get("ret", {}).get("mejor", "No entrenado"),
            ],
        }
    )
    data_table(resumen, 380)
    st.download_button(
        "⬇️ Descargar resumen (CSV)",
        resumen.to_csv(index=False).encode("utf-8-sig"),
        "resumen_sesion.csv",
        mime="text/csv",
    )

st.markdown(
    '<div class="footer">ML Analytics IA · Datos procesados localmente · '
    "Las métricas con datos simulados no representan resultados reales.</div>",
    unsafe_allow_html=True,
)
