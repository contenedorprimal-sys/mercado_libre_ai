"""Uso: python scripts/train.py data/mi_reporte.xlsx [data/costos.csv]
costos.csv: columnas 'sku','costo_unitario'."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_processing as dp, profitability_model as pm, returns_model as rm

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
    print(p["importancia"].head(8).round(3).to_string(index=False))
else:
    print(p["razon"])

r = rm.train(df)
print("\n== DEVOLUCIONES ==")
if r["viable"]:
    print(f"train={r['n_train']} test={r['n_test']} tasa_train={r['tasa_train']:.1%} tasa_test={r['tasa_test']:.1%}")
    print(r["metrics"].round(3).to_string(index=False))
    print(r["importancia"].head(8).round(3).to_string(index=False))
else:
    print(r["razon"])
