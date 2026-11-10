"""Recomendaciones basadas en datos. Describen asociaciones, no causas."""
import numpy as np
import pandas as pd
from .returns_model import build_target


def recommend(d: pd.DataFrame, min_n=10) -> pd.DataFrame:
    """d debe venir de add_economics()."""
    out = []
    if "sku" not in d or "ingresos_productos" not in d:
        return pd.DataFrame(columns=["tipo", "elemento", "evidencia"])
    g = d.groupby("sku").agg(ventas=("sku", "size"), resultado=("resultado_economico", "sum"),
                             ingresos=("ingresos_productos", "sum"))
    if "costos_envio" in d:
        g["envio"] = d.groupby("sku")["costos_envio"].apply(lambda s: s.abs().sum())
    g = g[g.ventas >= min_n].copy()
    if len(g):
        g["margen"] = g.resultado / g.ingresos.replace(0, np.nan)
        glob = d.resultado_economico.sum() / d.ingresos_productos.sum()
        for sku, r in g[g.margen < min(glob, g.margen.quantile(.25))].iterrows():
            out.append(("Revisar precio/costos", sku, f"Margen {r.margen:.1%} vs {glob:.1%} global ({int(r.ventas)} ventas)."))
        hi = g[(g.ingresos >= g.ingresos.quantile(.75)) & (g.margen <= g.margen.quantile(.25))]
        for sku, r in hi.iterrows():
            out.append(("Ingreso alto, rentabilidad baja", sku, f"Ingresos {r.ingresos:,.0f} MXN con margen {r.margen:.1%}."))
        if "envio" in g:
            g["envio_pct"] = g.envio / g.ingresos.replace(0, np.nan)
            for sku, r in g[g.envio_pct > g.envio_pct.median() * 1.5].iterrows():
                out.append(("Costo de envío elevado", sku, f"Envío = {r.envio_pct:.1%} del ingreso (mediana {g.envio_pct.median():.1%})."))
    if "fecha_venta" in d:
        v = d.assign(y=build_target(d)).dropna(subset=["y"])
        if len(v) and v.y.sum() > 0:
            base = v.y.mean()
            for col, tipo in [("sku", "Devoluciones frecuentes"), ("transportista", "Transportista con más devoluciones")]:
                if col in v:
                    t = v.groupby(col).y.agg(["size", "sum", "mean"])
                    for k, r in t[(t["size"] >= min_n) & (t["mean"] > 1.5 * base) & (t["sum"] >= 3)].iterrows():
                        out.append((tipo, k, f"{int(r['sum'])} devoluciones de {int(r['size'])} ({r['mean']:.1%}) vs {base:.1%} general. Asociación, no causa."))
    return pd.DataFrame(out, columns=["tipo", "elemento", "evidencia"])
