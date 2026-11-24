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
# Configuración general y constantes
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="ML Analytics IA · Mercado Libre",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="auto",
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

ICONS = {
    "marca": '<path d="M4 18.5V11"/><path d="M9.3 18.5V6.5"/><path d="M14.7 18.5v-6"/>'
             '<path d="M20 18.5V4.5"/><path d="M2.5 21h19"/>',
    "datos": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.66 3.13 3 7 3s7-1.34 7-3V6"/>'
             '<path d="M5 12v6c0 1.66 3.13 3 7 3s7-1.34 7-3v-6"/>',
    "carga": '<path d="M12 15.5V4"/><path d="m7.5 8.5 4.5-4.5 4.5 4.5"/>'
             '<path d="M4.5 14.5v3.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3.5"/>',
    "exploratorio": '<path d="M4.5 19.5V11"/><path d="M9.8 19.5V6.5"/><path d="M15.2 19.5v-5.5"/>'
                    '<path d="M20.5 19.5V4.5"/>',
    "rentabilidad": '<path d="M4 16.5 9.5 11l3.5 3.5L20 7"/><path d="M15 7h5v5"/>',
    "devoluciones": '<path d="M9.5 5.5 4.5 10.5l5 5"/><path d="M4.5 10.5H15a4.5 4.5 0 0 1 0 9h-4"/>',
    "comparacion": '<path d="M12 4v16"/><path d="M7.5 20h9"/><path d="M4 8.5h16"/>'
                   '<path d="M4 8.5 1.8 13.2a2.6 2.6 0 0 0 4.4 0Z"/>'
                   '<path d="M20 8.5 17.8 13.2a2.6 2.6 0 0 0 4.4 0Z"/>',
    "recomendaciones": '<path d="M9.5 18.5h5"/><path d="M10 21h4"/>'
                       '<path d="M12 3.5a5.5 5.5 0 0 0-3.2 9.95c.6.45 1 1.15 1 1.9v.15h4.4v-.15c0-.75.4-1.45 1-1.9A5.5 5.5 0 0 0 12 3.5Z"/>',
    "reportes": '<path d="M6.5 3.5h7l4.5 4.5v12a1 1 0 0 1-1 1h-10.5a1 1 0 0 1-1-1v-15a1 1 0 0 1 1-1Z"/>'
                '<path d="M13.5 3.5V8h4.5"/><path d="M8.5 13h7"/><path d="M8.5 16.5h7"/>',
    "privacidad": '<path d="M12 3.5 5 6v5.5c0 4.2 2.9 7.6 7 9 4.1-1.4 7-4.8 7-9V6Z"/>'
                  '<path d="m9 12 2.2 2.2L15.5 10"/>',
    "validacion": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5.2l3.3 2"/>',
    "modelos": '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/>'
               '<rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
}

S = st.session_state


def _svg(key: str, size: int = 24) -> str:
    body = ICONS.get(key, "")
    if not body:
        return ""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
  :root {
    --texto:#0B2545;
    --texto-2:#45556B;
    --texto-3:#55677C;
    --acento:#1E9E5A;
    --acento-osc:#157A47;
    --linea:#E3ECE6;
    --linea-2:#D7E4DC;
    --radio:14px;
    --fuente:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",
             Roboto,"Helvetica Neue",Arial,sans-serif;
  }

  html body, html body *:not([data-testid="stIconMaterial"]) { font-family:var(--fuente) !important; }
  [data-testid="stIconMaterial"] { font-family:"Material Symbols Rounded" !important; }

  .stApp { background: linear-gradient(180deg,#F5F8F6 0%,#EEF4F0 100%); }
  #MainMenu, footer { visibility: hidden; }
  [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] { display:none !important; }

  .stApp p { font-size:15px; line-height:1.5; }
  .stApp h1 { font-size:1.875rem !important; font-weight:700 !important;
              letter-spacing:-.02em !important; line-height:1.2 !important;
              color:var(--texto) !important; }
  .stApp h2 { font-size:1.375rem !important; font-weight:700 !important;
              letter-spacing:-.01em !important; color:var(--texto) !important; }
  .stApp h3 { font-size:1.125rem !important; font-weight:700 !important; color:var(--texto) !important; }
  .stApp h4 { font-size:.9375rem !important; font-weight:700 !important; color:var(--texto) !important; }
  [data-testid="stCaptionContainer"] p { font-size:13.5px !important; }
  [data-testid="stWidgetLabel"] p { font-size:13.5px !important; font-weight:600 !important; }
  [data-testid="stMetricLabel"] p { font-size:13px !important; font-weight:600 !important;
                                    letter-spacing:.02em !important; }
  [data-testid="stMetricValue"] p { font-size:1.45rem !important; font-weight:700 !important;
                                    font-variant-numeric:tabular-nums; letter-spacing:-.01em !important; }
  [data-testid="stTab"] p { font-size:15px !important; font-weight:600 !important; }
  [data-testid="stRadioOption"] p { font-size:15px !important; }

  [data-testid="stSidebar"] {
      background: linear-gradient(180deg,#0B2545 0%,#13315C 65%,#0E3B2E 100%);
      border-right: 1px solid rgba(255,255,255,.08);
      color:#EAF2EE;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
  [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] li, [data-testid="stSidebar"] a,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#EAF2EE; }
  [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5,
  [data-testid="stSidebar"] h6 { color:#8FE3B5 !important;
                                 letter-spacing:.1em; text-transform:uppercase;
                                 font-size:.8125rem !important; }
  [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.16); }
  [data-testid="stSidebar"] [data-testid="stHeaderActionElements"] a { color:rgba(234,242,238,.75); }

  [data-testid="stSidebar"] [data-testid="stRadioOption"] {
      border-radius:10px; padding:9px 10px !important;
      transition:background-color .15s ease;
  }
  [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background:rgba(255,255,255,.08); }
  [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
      background:rgba(255,255,255,.13);
      box-shadow:inset 3px 0 0 #35C77E;
  }
  [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
      font-weight:650 !important; color:#FFFFFF !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadioOption"]:focus-visible {
      outline:2px solid #8FE3B5; outline-offset:1px;
  }
  [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display:none; }

  .brand { display:flex; gap:.7rem; align-items:center; margin:.2rem 0 1.1rem; }
  .brand .logo {
      width:44px;height:44px;border-radius:12px;display:grid;place-items:center;
      background:linear-gradient(135deg,#157A47,#2EC27E); color:#fff;
      box-shadow:0 6px 18px rgba(0,0,0,.28);
  }
  .brand b { display:block; font-size:1.0625rem !important; line-height:1.2; color:#fff !important; }
  .brand span { font-size:.8125rem !important; color:#A9C4D6 !important;
                letter-spacing:.08em; text-transform:uppercase;
                display:flex; align-items:center; gap:.4rem; }
  .brand span svg { width:15px; height:15px; flex:none; opacity:.9; }
  .side-card b svg { width:16px; height:16px; vertical-align:-3px; margin-right:.4rem; }
  .pill em { font-style:normal; display:inline-flex; align-items:center;
             gap:.4rem; color:#8FE3B5; }
  .pill em svg { width:14px; height:14px; }
  .side-card {
      background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
      border-radius:12px; padding:.8rem .9rem; margin:.35rem 0 .6rem;
      font-size:.875rem; line-height:1.5; color:#D7E7DF !important;
  }
  .side-card b { display:block; color:#fff !important; font-size:.9375rem !important;
                 margin-bottom:.35rem; }
  .side-card .kv { display:flex; justify-content:space-between; gap:.75rem;
                   padding:.14rem 0; font-size:.8125rem !important; }
  .side-card .kv > span:first-child { color:#A9C4D6 !important; }
  .side-card .kv > span:last-child { color:#EAF2EE !important; text-align:right; }
  .dot { display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;
         flex:none; }
  .dot.on { background:#35C77E; box-shadow:0 0 0 3px rgba(53,199,126,.2); }
  .dot.off { background:#F4A261; }

  .hero {
      position:relative; overflow:hidden; border-radius:18px;
      padding:clamp(1.5rem,3vw,2.1rem) clamp(1.25rem,3vw,2.25rem);
      background:linear-gradient(125deg,#0B2545 0%,#123A63 58%,#0E3B2E 100%);
      color:#fff; margin-bottom:1.25rem; box-shadow:0 14px 34px rgba(11,37,69,.16);
      font-size:.9375rem;
  }
  .hero .kicker { font-size:.8125rem !important; letter-spacing:.14em;
                  text-transform:uppercase; color:#8FE3B5 !important; font-weight:600; }
  .hero h1 { color:#fff !important; font-size:clamp(1.6rem,2.4vw,2rem) !important;
             line-height:1.15; margin:.5rem 0 .6rem; }
  .hero p { color:#C8D9D2 !important; max-width:72ch; margin:0 0 1.05rem;
            font-size:.9375rem !important; }
  .pill {
      display:inline-block; padding:.32rem .72rem; margin:.2rem .35rem .2rem 0;
      border-radius:999px; background:rgba(255,255,255,.10);
      border:1px solid rgba(255,255,255,.20);
      font-size:.8125rem !important; color:#EAF2EE !important;
  }

  .page-head { display:flex; align-items:center; gap:.85rem; margin:.2rem 0 1.15rem; }
  .page-head > div:last-child { min-width:0; }
  .page-head .icon {
      width:46px;height:46px;flex:none;border-radius:13px;display:grid;place-items:center;
      background:#E4F4EA; border:1px solid #C9E8D5; color:var(--acento-osc);
  }
  .page-head .icon svg { display:block; }
  .page-head h1 { margin:0 !important; font-size:1.75rem !important; }
  .page-head p { margin:0 !important; color:var(--texto-2) !important;
                 font-size:.9375rem !important; }

  div[data-testid="stMetric"] {
      background:#fff; border:1px solid var(--linea);
      border-radius:var(--radio); padding:.9rem 1rem;
      box-shadow:0 1px 2px rgba(11,37,69,.05);
  }
  div[data-testid="stMetric"] label { color:var(--texto-2) !important; }

  .stTabs [role="tablist"] { gap:.15rem; border-bottom:1px solid var(--linea); }
  .stTabs [data-testid="stTab"] {
      padding:.8rem .7rem !important; border-radius:9px 9px 0 0;
      color:var(--texto-2) !important; transition:color .15s ease;
  }
  .stTabs [data-testid="stTab"]:hover { color:var(--texto) !important; }
  .stTabs [data-testid="stTab"][aria-selected="true"] { color:var(--texto) !important; }
  .stTabs [data-testid="stTab"] .react-aria-SelectionIndicator {
      height:3px !important; top:auto !important; bottom:-1px !important;
      border-radius:999px !important; background:var(--acento-osc) !important;
  }
  .stTabs [data-testid="stTab"]:focus-visible {
      outline:2px solid var(--acento-osc); outline-offset:-2px;
  }

  .stButton > button, .stDownloadButton > button {
      border-radius:10px !important; font-weight:600 !important;
      font-size:.9375rem !important; border:1px solid var(--linea-2) !important;
      background:#fff !important; color:var(--texto) !important;
      transition:background-color .15s ease, border-color .15s ease;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
      background:#F1F6F3 !important; border-color:#BCD2C6 !important;
  }
  button[kind="primary"] {
      background:var(--acento-osc) !important; background-image:none !important;
      border-color:var(--acento-osc) !important; color:#fff !important;
      box-shadow:0 4px 12px rgba(21,122,71,.24);
  }
  button[kind="primary"]:hover {
      background:#11693C !important; border-color:#11693C !important; color:#fff !important;
  }
  :is(.stButton,.stDownloadButton) > button:focus-visible, button[kind="primary"]:focus-visible {
      outline:2px solid var(--acento-osc) !important; outline-offset:2px !important;
  }
  button:disabled { opacity:.55; }

  div[data-testid="stDataFrame"] { border:1px solid var(--linea); border-radius:12px; overflow:hidden; }
  .nota {
      background:#FFF8EC; border:1px solid #F4D9A6; border-left:4px solid #D9822B;
      border-radius:12px; padding:.8rem 1rem; color:#6B4E1F !important;
      font-size:.9375rem !important; line-height:1.5 !important;
  }
  .ok {
      background:#EAF8F0; border:1px solid #BFE8CE; border-left:4px solid var(--acento-osc);
      border-radius:12px; padding:.8rem 1rem; color:#14532D !important;
      font-size:.9375rem !important; line-height:1.5 !important;
  }
  .step { background:#fff; border:1px solid var(--linea); border-radius:14px;
          padding:1.05rem 1.1rem; height:100%; box-shadow:0 1px 2px rgba(11,37,69,.05); }
  .step .n {
      width:30px;height:30px;border-radius:9px;display:grid;place-items:center;
      background:#E4F4EA;color:var(--acento-osc);font-weight:700;font-size:.9375rem;
      margin-bottom:.6rem;
  }
  .step b { color:var(--texto) !important; font-size:1rem !important; }
  .step p { color:var(--texto-2) !important; font-size:.875rem !important; margin:.3rem 0 0; }
  .footer { text-align:center; color:var(--texto-3) !important;
            font-size:.8125rem !important; padding:2rem 0 .75rem; }

  [data-testid="stColumn"] { min-width:200px; }
  [data-testid="stMainBlockContainer"] { max-width:1460px; margin-inline:auto; }
  @media (max-width:700px) {
      [data-testid="stMainBlockContainer"] { padding:52px 20px 96px !important; }
      [data-testid="stColumn"] { min-width:100% !important; }
      .page-head { align-items:flex-start; }
      .page-head h1 { font-size:1.5rem !important; }
  }

  @media (prefers-reduced-motion:reduce) {
      *, *::before, *::after {
          animation-duration:.01ms !important; animation-iteration-count:1 !important;
          transition-duration:.01ms !important; scroll-behavior:auto !important;
      }
  }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Utilidades de persistencia
# --------------------------------------------------------------------------- #
def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return cleaned[:80] or "archivo"


def save_dataset(df: pd.DataFrame, costs: pd.DataFrame | None, source: str) -> None:
    """Guarda el dataset limpio, catálogo de costos y metadatos en disco."""
    try:
        CURRENT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CURRENT_DIR / "dataset.csv", index=False)
        costs_path = CURRENT_DIR / "costs.csv"
        if costs is not None and len(costs):
            costs.to_csv(costs_path, index=False)
        elif costs_path.exists():
            costs_path.unlink()
        meta = {
            "source": _safe_name(source),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "log": list(S.get("log", [])),
        }
        (CURRENT_DIR / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        st.warning(f"No se pudo guardar la sesión en disco: {exc}")


def load_saved() -> tuple[pd.DataFrame, pd.DataFrame | None, dict] | None:
    """Recupera la última sesión guardada (datos + costos + metadatos)."""
    dataset_path = CURRENT_DIR / "dataset.csv"
    if not dataset_path.exists():
        return None
    try:
        df = pd.read_csv(dataset_path)
        for col in getattr(dp, "DATES", ()):
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        costs_path = CURRENT_DIR / "costs.csv"
        costs = pd.read_csv(costs_path) if costs_path.exists() else None
        meta_path = CURRENT_DIR / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return df, costs, meta
    except Exception:
        return None


def clear_saved() -> None:
    for name in ("dataset.csv", "costs.csv", "meta.json"):
        path = CURRENT_DIR / name
        if path.exists():
            path.unlink()


def model_on_disk(kind: str) -> Path:
    return MODELS_DIR / MODEL_FILES[kind]


def load_model_from_disk(kind: str) -> dict | None:
    path = model_on_disk(kind)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        return {
            "viable": True,
            "from_disk": True,
            "mejor": bundle.get("nombre", "modelo guardado"),
            "metrics": bundle.get("metrics"),
            "umbrales": bundle.get("umbrales"),
            "tipo_resultado": bundle.get("tipo_resultado"),
        }
    except Exception:
        return None


def hydrate_once() -> None:
    """Recupera sesión y modelos persistidos una sola vez al iniciar."""
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
            df, log = dp.clean(dp.load_file(DEMO_FILE))
            S["df"], S["log"], S["source"], S["demo"] = df, log, DEMO_FILE.name, True
        except Exception:
            pass

    for kind in ("prof", "ret"):
        if kind not in S:
            loaded = load_model_from_disk(kind)
            if loaded:
                S[kind] = loaded


hydrate_once()


# --------------------------------------------------------------------------- #
# Helpers de datos, métricas y UI
# --------------------------------------------------------------------------- #
def page_head(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-head"><div class="icon">{_svg(icon, 24)}</div>'
        f"<div><h1>{title}</h1><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def hero() -> None:
    pills = "".join(
        f'<span class="pill"><em>{_svg(k, 14)}</em>{label}</span>'
        for k, label in (
            ("privacidad", "Sin datos personales"),
            ("validacion", "Validación temporal"),
            ("modelos", "Modelos comparados"),
        )
    )
    st.markdown(
        '<div class="hero">'
        '<div class="kicker">Mercado Libre · Analítica con IA</div>'
        "<h1>Rentabilidad y riesgo de devoluciones,<br>antes de que sea un problema</h1>"
        "<p>Sube tu reporte de ventas y obtén indicadores económicos, predicciones con "
        "validación temporal, factores asociados y recomendaciones accionables. "
        "Todo el procesamiento ocurre de forma local y sin datos personales.</p>"
        f"{pills}"
        "</div>",
        unsafe_allow_html=True,
    )


def money(x: float) -> str:
    if x is None or pd.isna(x):
        return "N/D"
    sign = "\u2212" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "N/D"
    sign = "\u2212" if x < 0 else ""
    return f"{sign}{abs(x):.1%}"


def chart(fig: go.Figure, height: int | None = None) -> None:
    fig.update_layout(
        template="plotly_white",
        colorway=COLORES,
        margin=dict(l=10, r=10, t=52, b=64),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=AZUL, size=13),
        title=dict(font=dict(size=15, color=AZUL), x=0.01),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(tickfont=dict(size=13, color=AZUL), title_font=dict(size=13, color=AZUL))
    fig.update_yaxes(tickfont=dict(size=13, color=AZUL), title_font=dict(size=13, color=AZUL))
    fig.update_layout(legend_font=dict(size=13, color=AZUL))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def data_table(df: pd.DataFrame, height: int | None = None) -> None:
    kwargs = {}
    if height:
        kwargs["height"] = max(96, min(height, 38 + max(len(df), 1) * 35))
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


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
    """Calcula economía y aplica filtros globales de fecha y SKU."""
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
            picked = st.multiselect("SKU", skus, placeholder="Todos los SKU")
            if picked:
                d = d[d.sku.astype(str).isin(picked)]
        st.caption(f"{len(d):,} de {len(S['df']):,} ventas en el filtro")
    return d


def _summary_stats(d: pd.DataFrame) -> dict:
    parcial = bool(d["tipo_resultado"].ne("estimado con costo").any()) if "tipo_resultado" in d else True
    dg = rm.diagnose(d)
    ver = dg.get("devoluciones(1)", 0) + dg.get("no_devoluciones(0)", 0)
    tasa_dev = dg.get("devoluciones(1)", 0) / ver if ver else np.nan
    ingresos = d["ingresos_productos"].sum() if "ingresos_productos" in d else np.nan
    resultado = d["resultado_economico"].sum() if "resultado_economico" in d else np.nan
    return {
        "parcial": parcial,
        "dg": dg,
        "ver": ver,
        "tasa_dev": tasa_dev,
        "ingresos": ingresos,
        "resultado": resultado,
    }


def _top_sku_bar(g: pd.DataFrame, metric: str, title: str, color: str, height: int) -> None:
    top = g.nlargest(10, metric).sort_values(metric)
    chart(
        px.bar(top, x=metric, y="sku", orientation="h", title=title, color_discrete_sequence=[color]),
        height,
    )


def _best_metric_row(result: dict) -> pd.Series | None:
    metrics = result.get("metrics")
    if metrics is None or not len(metrics) or "Modelo" not in metrics:
        return None
    matches = metrics[metrics["Modelo"] == result.get("mejor")]
    return matches.iloc[0] if len(matches) else metrics.iloc[0]


def load_demo() -> bool:
    if not DEMO_FILE.exists():
        st.error("No se encontró `data/demo_simulado.csv`.")
        return False
    df, log = dp.clean(dp.load_file(DEMO_FILE))
    S["df"], S["log"], S["source"], S["demo"] = df, log, DEMO_FILE.name, True
    S.pop("prof", None)
    S.pop("ret", None)
    save_dataset(df, S.get("costs"), DEMO_FILE.name)
    return True


def _run_training(kind: str) -> None:
    is_prof = kind == "prof"
    spinner_msg = (
        "Entrenando y comparando modelos de rentabilidad…"
        if is_prof
        else "Entrenando y comparando modelos de devoluciones…"
    )
    with st.spinner(spinner_msg):
        res = (
            pm.train(S["df"], S.get("costs"), models_dir=str(MODELS_DIR))
            if is_prof
            else rm.train(S["df"], models_dir=str(MODELS_DIR))
        )
    S[kind] = res
    if res.get("viable"):
        etiqueta = "rentabilidad" if is_prof else "devoluciones"
        st.toast(f"Modelo de {etiqueta} listo: {res['mejor']}")
    else:
        st.toast("El modelo no es viable con estos datos. Revisa el volumen de datos.")


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
        '<div class="brand"><div class="logo">'
        + _svg("marca", 26)
        + '</div><div><b>ML Analytics</b><span>'
        + _svg("datos", 15)
        + "Mercado Libre · IA</span></div></div>",
        unsafe_allow_html=True,
    )
    sec = st.radio("Menú", SECCIONES, key="menu")
    st.divider()

    if "df" in S:
        df0 = S["df"]
        rango = ""
        if "fecha_venta" in df0 and df0.fecha_venta.notna().any():
            rango = f"{df0.fecha_venta.min():%d/%m/%Y} – {df0.fecha_venta.max():%d/%m/%Y}"
        n_skus = df0.sku.nunique() if "sku" in df0 else "N/D"
        st.markdown(
            '<div class="side-card"><b>Conjunto activo</b>'
            f'<div class="kv"><span>Archivo</span><span>{S.get("source", "datos")}</span></div>'
            f'<div class="kv"><span>Ventas</span><span>{len(df0):,}</span></div>'
            f'<div class="kv"><span>SKUs</span><span>{n_skus}</span></div>'
            f'<div class="kv"><span>Periodo</span><span>{rango or "sin fechas"}</span></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="side-card"><b>Sin datos cargados</b><br>'
            "Sube tu reporte o usa el conjunto de demostración.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="side-card"><b>Modelos</b>'
        f'<div class="kv"><span>Rentabilidad</span><span>{model_badge("prof")}</span></div>'
        f'<div class="kv"><span>Devoluciones</span><span>{model_badge("ret")}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("El archivo original nunca se modifica. Datos personales excluidos.")


# --------------------------------------------------------------------------- #
# Vistas
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
        if a.button("Ir a Carga de datos", type="primary", width="stretch"):
            S["menu"] = "Carga de datos"
            st.rerun()
        if b.button("Explorar con datos de demostración", width="stretch"):
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

    stats = _summary_stats(d)
    dg = stats["dg"]
    c = st.columns(5)
    c[0].metric("Ventas", f"{len(d):,}", help="Filas del reporte dentro del filtro actual.")
    c[1].metric(
        "Ingresos por producto",
        money(stats["ingresos"]),
        help="Ingresos brutos por productos antes de cargos, envíos y reembolsos.",
    )
    c[2].metric(
        "Resultado PARCIAL" if stats["parcial"] else "Resultado estimado",
        money(stats["resultado"]),
        help="Total de Mercado Libre menos costo de adquisición (si existe catálogo).",
    )
    c[3].metric("SKUs activos", d.sku.nunique() if "sku" in d else "N/D")
    c[4].metric(
        "Tasa de devolución",
        pct(stats["tasa_dev"]),
        help=f"{dg.get('devoluciones(1)', 0)} devoluciones verificadas sobre {stats['ver']} ventas con resultado verificable.",
    )

    if stats["parcial"]:
        st.markdown(
            '<div class="nota">Sin catálogo de costos el resultado es <b>parcial</b> '
            "(no es utilidad neta). Cárgalo en <i>Carga de datos</i> para obtener el resultado estimado.</div>",
            unsafe_allow_html=True,
        )

    t1, t2, t3 = st.tabs(["Evolución mensual", "Top SKUs", "Actividad reciente"])

    with t1:
        if "fecha_venta" in d and d.fecha_venta.notna().any() and "resultado_economico" in d:
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
            st.info("El conjunto no tiene fechas de venta válidas o columna de resultado económico.")

    with t2:
        if "sku" in d and "resultado_economico" in d:
            g = (
                d.groupby("sku")
                .agg(
                    ventas=("id_venta", "count"),
                    ingresos=("ingresos_productos", "sum") if "ingresos_productos" in d else ("id_venta", "count"),
                    resultado=("resultado_economico", "sum"),
                )
                .reset_index()
            )
            a, b = st.columns(2)
            with a:
                _top_sku_bar(g, "ventas", "SKUs con más ventas", VERDE, 360)
            with b:
                _top_sku_bar(g, "resultado", "SKUs con mayor resultado", AZUL_2, 360)
        else:
            st.info("El conjunto no incluye las columnas requeridas de SKU y resultado económico.")

    with t3:
        cols = [
            col for col in [
                "id_venta", "fecha_venta", "sku", "estado", "unidades",
                "ingresos_productos", "resultado_economico", "tipo_resultado",
            ]
            if col in d
        ]
        recent = d[cols].sort_values("fecha_venta", ascending=False).head(25) if "fecha_venta" in d else d[cols].head(25)
        data_table(recent, 380)


elif sec == "Carga de datos":
    if S.pop("_flash", None):
        st.success("Datos y modelos borrados. Vuelve a cargar un reporte para empezar.")
    page_head("carga", "Carga de datos", "Sube tu reporte de Mercado Libre y, si lo tienes, tu catálogo de costos.")

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
                        df, log = dp.clean(dp.load_file(f))
                    S["df"], S["log"], S["source"] = df, log, f.name
                    for key in ("demo", "prof", "ret"):
                        S.pop(key, None)
                    S["_upload_sig"] = firma
                    save_dataset(df, S.get("costs"), f.name)
                    st.toast(f"{len(df):,} filas cargadas y guardadas")
                    st.rerun()
                except Exception as exc:
                    st.error(
                        f"No se pudo leer el archivo {f.name}. Revisa que sea un export "
                        f"de Mercado Libre en Excel o CSV. Detalle: {exc}"
                    )
                    S.pop("_upload_sig", None)

        if st.button("Usar datos de demostración", width="stretch"):
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
                    costs.columns = [str(col).strip().lower() for col in costs.columns]
                    if not {"sku", "costo_unitario"} <= set(costs.columns):
                        raise ValueError("El catálogo debe tener las columnas 'sku' y 'costo_unitario'.")
                    S["costs"] = costs
                    S["_costs_sig"] = firma
                    if "df" in S:
                        save_dataset(S["df"], costs, S.get("source", "datos"))
                    st.toast(f"{len(costs):,} costos cargados")
                    st.rerun()
                except Exception as exc:
                    st.error(
                        f"No se pudo leer el catálogo {cf.name}. Debe ser CSV o Excel con las "
                        f"columnas sku y costo_unitario. Detalle: {exc}"
                    )
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

    t1, t2, t3, t4 = st.tabs(["Limpieza", "Calidad", "Devoluciones", "Vista previa"])
    with t1:
        st.markdown("**Decisiones aplicadas**")
        for item in S.get("log", []):
            st.markdown(f"- {item}")
    with t2:
        a, b, c = st.columns(3)
        a.metric("Filas", f"{rep['filas']:,}")
        b.metric("Columnas", rep["columnas"])
        c.metric("IDs de venta repetidos", rep.get("id_venta_repetidos", "N/D"))
        x, y = st.columns(2)
        with x:
            st.write("**Inconsistencias**")
            inc = rep.get("inconsistencias", {})
            data_table(
                pd.DataFrame({"hallazgo": list(inc), "casos": list(inc.values())})
                if inc
                else pd.DataFrame({"hallazgo": ["Sin hallazgos"]})
            )
        with y:
            st.write("**Atípicos (IQR × 3)**")
            at = rep.get("atipicos_iqr3", {})
            data_table(
                pd.DataFrame({"columna": list(at), "casos": list(at.values())})
                if at
                else pd.DataFrame({"columna": ["Sin atípicos"]})
            )
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
            "Descargar datos procesados (CSV)",
            df.to_csv(index=False).encode("utf-8-sig"),
            "datos_procesados.csv",
            mime="text/csv",
        )

    with st.expander("Eliminar datos guardados"):
        st.markdown(
            '<div class="nota">Acción destructiva: borra el conjunto persistido en disco '
            "y los resultados de la sesión. No se puede deshacer.</div>",
            unsafe_allow_html=True,
        )
        confirmar = st.checkbox(
            "Entiendo que se borrarán los datos y los modelos de esta sesión.",
            key="confirmar_borrar",
        )
        if st.button(
            "Borrar datos y modelos",
            type="primary",
            disabled=not confirmar,
            width="stretch",
        ):
            clear_saved()
            for key in ("df", "costs", "log", "meta", "source", "prof", "ret", "_upload_sig", "_costs_sig", "demo"):
                S.pop(key, None)
            S["_flash"] = "Datos y modelos borrados. Vuelve a cargar un reporte para empezar."
            S.pop("confirmar_borrar", None)
            st.rerun()


elif sec == "Análisis exploratorio":
    page_head("exploratorio", "Análisis exploratorio", "Panorama de productos, envíos y evolución del negocio.")
    d = filtered()
    if not len(d):
        st.warning("El filtro actual no devuelve ventas.")
        st.stop()

    t1, t2, t3 = st.tabs(["Productos", "Envíos", "Importes"])

    with t1:
        if "sku" in d and "ingresos_productos" in d and "resultado_economico" in d:
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
                _top_sku_bar(g, "ventas", "Más vendidos (unidades de venta)", VERDE, 380)
            with b:
                _top_sku_bar(g, "ingresos", "Mayores ingresos por producto", AZUL_2, 380)
            st.write("**Detalle por SKU**")
            data_table(g, 320)
        else:
            st.info("El conjunto no incluye las columnas de SKU e importes requeridas.")

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
        if "resultado_economico" in d and "tipo_resultado" in d:
            chart(
                px.box(
                    d, x="tipo_resultado", y="resultado_economico",
                    title="Resultado económico por tipo de cálculo",
                    color="tipo_resultado", color_discrete_sequence=[VERDE, AMBAR],
                ),
                360,
            )
        cols = [
            col for col in [
                "ingresos_productos", "cargo_venta", "ingresos_envio",
                "costos_envio", "descuentos", "anulaciones", "resultado_economico",
            ]
            if col in d
        ]
        if cols:
            st.write("**Resumen estadístico (MXN)**")
            data_table(d[cols].describe().T.round(2), 300)


elif sec == "Rentabilidad":
    page_head("rentabilidad", "Rentabilidad", "Resultado económico por venta y modelos de regresión comparados.")
    d = filtered()

    izq, der = st.columns([1.2, 2])
    with izq:
        st.markdown("#### Entrenamiento")
        st.caption(
            "Se comparan baseline, regresión lineal, Random Forest y XGBoost. "
            "El 80 % más antiguo entrena y el 20 % más reciente prueba (validación temporal)."
        )
        if st.button(
            "Reentrenar modelos" if S.get("prof") else "Entrenar modelos",
            type="primary",
            width="stretch",
        ):
            _run_training("prof")
            st.rerun()
        if S.get("prof", {}).get("from_disk"):
            st.info("Modelo recuperado del disco. Reentrena para ver importancia y predicciones.")
    with der:
        r = S.get("prof")
        if r and r.get("viable"):
            mejor_row = _best_metric_row(r)
            a, b, c = st.columns(3)
            a.metric("Mejor modelo", r["mejor"])
            b.metric("RMSE (prueba)", f"{mejor_row['RMSE']:,.2f}" if mejor_row is not None and "RMSE" in mejor_row else "N/D")
            c.metric("R² (prueba)", f"{mejor_row['R2']:.3f}" if mejor_row is not None and "R2" in mejor_row else "N/D")
            if r.get("periodo_train") and r.get("periodo_test"):
                st.caption(
                    f"Entrenamiento: {r['periodo_train'][0]:%d/%m/%Y} → {r['periodo_train'][1]:%d/%m/%Y} · "
                    f"Prueba: {r['periodo_test'][0]:%d/%m/%Y} → {r['periodo_test'][1]:%d/%m/%Y}"
                )
            if r.get("metrics") is not None:
                data_table(r["metrics"].round(3).rename(columns={"R2": "R²"}))
        elif r and not r.get("viable"):
            st.error(r.get("razon", "Modelo no viable."))
        else:
            st.markdown(
                '<div class="nota">Aún no hay modelo de rentabilidad. Pulsa '
                "<b>Entrenar modelos</b> para comparar algoritmos.</div>",
                unsafe_allow_html=True,
            )

    r = S.get("prof")
    if r and r.get("viable") and r.get("importancia") is not None:
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
            preds = r.get("predicciones")
            if preds is not None and len(preds):
                lo = float(min(preds.real.min(), preds.predicho.min()))
                hi = float(max(preds.real.max(), preds.predicho.max()))
                fig = go.Figure()
                fig.add_scatter(
                    x=preds.real, y=preds.predicho, mode="markers", name="Ventas",
                    marker=dict(color=VERDE, size=7, opacity=.6),
                )
                fig.add_scatter(
                    x=[lo, hi], y=[lo, hi], mode="lines", name="Predicción perfecta",
                    line=dict(color=AZUL_2, dash="dash"),
                )
                fig.update_layout(
                    title="Real vs predicho (conjunto de prueba)",
                    xaxis_title="Resultado real (MXN)",
                    yaxis_title="Resultado predicho (MXN)",
                )
                chart(fig, 400)
            else:
                st.info("Reentrena el modelo para generar las predicciones de prueba.")

    if "sku" in d and "resultado_economico" in d and len(d):
        st.divider()
        st.markdown("#### Resultado económico por SKU")
        g = d.groupby("sku").resultado_economico.sum().sort_values()
        if len(g) <= 20:
            st.write("**Ranking completo**")
            data_table(g.reset_index().rename(columns={"resultado_economico": "resultado"}), 360)
        else:
            a, b = st.columns(2)
            with a:
                st.write("**Menor resultado**")
                data_table(g.head(10).reset_index().rename(columns={"resultado_economico": "resultado"}), 320)
            with b:
                st.write("**Mejor resultado**")
                data_table(g.tail(10)[::-1].reset_index().rename(columns={"resultado_economico": "resultado"}), 320)


elif sec == "Predicción de devoluciones":
    page_head("devoluciones", "Predicción de devoluciones", "Clasificación del riesgo relativo con evidencia verificable.")
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
        if st.button(
            "Reentrenar modelos" if S.get("ret") else "Entrenar modelos",
            type="primary",
            width="stretch",
        ):
            _run_training("ret")
            st.rerun()
        if S.get("ret", {}).get("from_disk"):
            st.info("Modelo recuperado del disco. Reentrena para ver importancia y umbrales nuevos.")
    with der:
        r = S.get("ret")
        if r and r.get("viable"):
            best = _best_metric_row(r)
            a, b, c = st.columns(3)
            a.metric("Mejor modelo", r["mejor"])
            b.metric("ROC-AUC", f"{best['ROC-AUC']:.3f}" if best is not None and "ROC-AUC" in best else "N/D")
            c.metric("F1", f"{best['F1']:.3f}" if best is not None and "F1" in best else "N/D")
            if r.get("metrics") is not None:
                data_table(r["metrics"].round(3))
        elif r and not r.get("viable"):
            st.error(r.get("razon", "Modelo no viable."))
        else:
            st.markdown(
                '<div class="nota">Aún no hay modelo de devoluciones. Pulsa '
                "<b>Entrenar modelos</b>.</div>",
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
            cols = [col for col in ["id_venta", "sku", "transportista", "prob_devolucion", "riesgo"] if col in out]
            tabla = out[cols].sort_values("prob_devolucion", ascending=False).head(100) if "prob_devolucion" in out else out[cols].head(100)
            if "prob_devolucion" in tabla:
                tabla["prob_devolucion"] = tabla["prob_devolucion"].round(3)
            data_table(tabla, 420)
            st.caption("Bajo / Medio / Alto se definen por percentiles de la probabilidad estimada.")
            st.download_button(
                "Descargar riesgo por venta (CSV)",
                out[cols].to_csv(index=False).encode("utf-8-sig"),
                "riesgo_devoluciones.csv",
                mime="text/csv",
            )


elif sec == "Comparación de modelos":
    page_head("comparacion", "Comparación de modelos", "Métricas lado a lado para elegir con criterio.")
    t1, t2 = st.tabs(["Rentabilidad", "Devoluciones"])

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


elif sec == "Recomendaciones inteligentes":
    page_head("recomendaciones", "Recomendaciones inteligentes", "Hallazgos accionables basados en los datos observados.")
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
            "Descargar recomendaciones (CSV)",
            out.to_csv(index=False).encode("utf-8-sig"),
            "recomendaciones.csv",
            mime="text/csv",
        )
    st.caption("Las recomendaciones describen asociaciones observadas; no demuestran causalidad.")


elif sec == "Reportes":
    page_head("reportes", "Reportes", "Exporta los resultados de la sesión actual.")
    d = filtered()

    st.markdown("#### Descargas")
    a, b, c = st.columns(3)
    with a:
        st.download_button(
            "Descargar datos con resultado económico",
            d.to_csv(index=False).encode("utf-8-sig"),
            "reporte_economico.csv",
            mime="text/csv",
            width="stretch",
        )
    with b:
        prof = S.get("prof")
        if prof and prof.get("viable") and prof.get("metrics") is not None:
            st.download_button(
                "Descargar métricas de rentabilidad",
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
                "Descargar métricas de devoluciones",
                ret["metrics"].to_csv(index=False).encode("utf-8-sig"),
                "metricas_devoluciones.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("Métricas de devoluciones", disabled=True, width="stretch")

    st.divider()
    st.markdown("#### Resumen de la sesión")
    stats = _summary_stats(d)
    dg = stats["dg"]
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
                money(stats["ingresos"]),
                money(stats["resultado"]),
                "PARCIAL (sin costos)" if stats["parcial"] else "Estimado con costo",
                f"{dg.get('devoluciones(1)', 0):,}",
                pct(stats["tasa_dev"]),
                S.get("prof", {}).get("mejor", "No entrenado"),
                S.get("ret", {}).get("mejor", "No entrenado"),
            ],
        }
    )
    data_table(resumen, 380)
    st.download_button(
        "Descargar resumen (CSV)",
        resumen.to_csv(index=False).encode("utf-8-sig"),
        "resumen_sesion.csv",
        mime="text/csv",
    )

st.markdown(
    '<div class="footer">ML Analytics IA · Datos procesados localmente · '
    "Las métricas con datos simulados no representan resultados reales.</div>",
    unsafe_allow_html=True,
)
