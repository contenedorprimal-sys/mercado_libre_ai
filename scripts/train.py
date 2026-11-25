"""Uso: python scripts/train.py data/mi_reporte.xlsx [data/costos.csv]

costos.csv: columnas 'sku','costo_unitario'.
Entrena y compara los modelos supervisados, añade segmentación no supervisada
y muestra el criterio de selección y el análisis multivariado exploratorio.
"""
import sys
from pathlib import Path

import pandas as pd

# Debe ir ANTES de importar `src` cuando el script se ejecuta desde scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_processing as dp                      # noqa: E402
from src import multivariate as mv                         # noqa: E402
from src import profitability_model as pm                  # noqa: E402
from src import returns_model as rm                        # noqa: E402
from src import unsupervised_model as um                   # noqa: E402


def _print_selection(title: str, result: dict) -> None:
    seleccion = result.get("seleccion") or {}
    if not seleccion:
        return
    print(f"\nCriterio de selección ({title}): {seleccion.get('reason')}")
    ranking = seleccion.get("ranking")
    if ranking is not None and len(ranking):
        print(ranking.round(3).to_string(index=False))


if len(sys.argv) < 2:
    raise SystemExit("Uso: python scripts/train.py data/mi_reporte.xlsx [data/costos.csv]")
raw = dp.load_file(sys.argv[1])
df, log = dp.clean(raw)
costs = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
print("Columnas detectadas:", list(df.columns), "\n")
print("\n".join(log), "\n")
rep = dp.quality_report(df)
print("Calidad:", {k: v for k, v in rep.items() if k != "faltantes_pct"}, "\n")
print("== DIAGNÓSTICO DEL OBJETIVO DE DEVOLUCIONES ==\n", rm.diagnose(df), "\n")

p = pm.train(df, costs)
print("== RENTABILIDAD ==")
if p["viable"]:
    print(p["tipo_resultado"], f"train={p['n_train']} test={p['n_test']}")
    print(p["metrics"].round(3).to_string(index=False))
    print("Mejor modelo:", p["mejor"])
    print(p["importancia"].head(8).round(3).to_string(index=False))
    _print_selection("rentabilidad", p)
    if p.get("advertencia"):
        print("ADVERTENCIA:", p["advertencia"])
else:
    print(p["razon"])

r = rm.train(df)
print("\n== DEVOLUCIONES ==")
if r["viable"]:
    print(f"train={r['n_train']} test={r['n_test']} tasa_train={r['tasa_train']:.1%} "
          f"tasa_test={r['tasa_test']:.1%}")
    print(r["metrics"].round(3).to_string(index=False))
    print("Mejor modelo:", r["mejor"])
    print(r["importancia"].head(8).round(3).to_string(index=False))
    _print_selection("devoluciones", r)
    if r.get("advertencia"):
        print("ADVERTENCIA:", r["advertencia"])
else:
    print(r["razon"])

u = um.fit_unsupervised(df)
print("\n== NO SUPERVISADO (KMEANS + ISOLATION FOREST) ==")
if u["viable"]:
    print(f"registros={u['n_registros']} k={u['best_k']} silueta={u['silueta']} "
          f"scores={u['silhouette_scores']}")
    print(f"anomalías={u['n_anomalias']} ({u['pct_anomalias']}%)")
    print(u["resumen_clusters"].round(3).to_string(index=False))
    saved_path = um.save_model(u)
    print(f"Modelo guardado en: {saved_path}")
else:
    print(u["razon"])

multi = u.get("multivariado") or mv.multivariate_report(df)
print("\n== ANÁLISIS MULTIVARIADO ==")
if multi.get("aplica"):
    pca = multi["pca"]
    print("Variables:", ", ".join(multi["variables"]))
    print("Varianza explicada (PC1..):", pca["varianza_explicada"])
    print("Componentes para 80%:", pca["n_componentes_80"])
    print("\nVIF:")
    print(multi["vif"].to_string(index=False))
    print("\nCorrelación:")
    print(multi["correlacion"].to_string())
    print("\nAplicabilidad:", multi["razon"])
else:
    print(multi["razon"])
