"""Política única de selección de modelos (regla de dominio compartida).

Antes cada modelo elegía su "mejor" de forma distinta y sin comprobar que
superara al baseline. Esta función concentra el criterio para que CLI, UI y
pruebas usen exactamente la misma regla:

* Regresión: menor RMSE; se exige una mejora relativa mínima frente al baseline.
* Clasificación: mayor ROC-AUC; se exige una mejora absoluta mínima.

La métrica se calcula siempre sobre el conjunto de prueba temporal (futuro),
nunca sobre entrenamiento. Si ningún candidato supera al baseline, se devuelve
`confiable=False` y el motivo, pero el modelo se conserva para no romper el
flujo: el sistema informa que no hay evidencia de mejora real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TASKS = {
    "regression": {
        "metric": "RMSE",
        "higher_is_better": False,
        # 1 % de mejora relativa evita elegir diferencias que son ruido.
        "min_improvement": 0.01,
        "relative": True,
    },
    "classification": {
        "metric": "ROC-AUC",
        "higher_is_better": True,
        # 0.01 de ROC-AUC es el mínimo práctico frente a un baseline.
        "min_improvement": 0.01,
        "relative": False,
    },
}


def _score(row: pd.Series, metric: str) -> float:
    try:
        return float(row[metric])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def choose_model(
    metrics: pd.DataFrame,
    task: str,
    baseline_prefix: str = "Baseline",
) -> dict:
    """Elige el mejor modelo no-baseline y comprueba que supere al baseline.

    Devuelve un diccionario serializable con la decisión y su justificación.
    No lanza excepción si faltan filas: devuelve ``best=None``.
    """
    if task not in TASKS:
        raise ValueError(f"Tarea desconocida: {task}. Usa {sorted(TASKS)}.")
    cfg = TASKS[task]
    metric = cfg["metric"]
    if metrics is None or not len(metrics) or "Modelo" not in metrics:
        return {"best": None, "metric": metric, "reason": "Sin tabla de métricas."}

    named = metrics[metrics["Modelo"].notna()].copy()
    is_base = named["Modelo"].astype(str).str.startswith(baseline_prefix)
    candidates = named[~is_base].copy()
    candidates["_score"] = candidates.apply(lambda r: _score(r, metric), axis=1)
    candidates = candidates[np.isfinite(candidates["_score"])]
    if not len(candidates):
        return {"best": None, "metric": metric, "reason": f"Ningún modelo tiene {metric} válido."}

    ranked = candidates.sort_values("_score", ascending=not cfg["higher_is_better"])
    best_row = ranked.iloc[0]
    best = str(best_row["Modelo"])
    best_score = float(best_row["_score"])

    base_rows = named[is_base].copy()
    base_score = float("nan")
    if len(base_rows):
        base_rows["_score"] = base_rows.apply(lambda r: _score(r, metric), axis=1)
        base_rows = base_rows[np.isfinite(base_rows["_score"])]
        if len(base_rows):
            base_score = float(base_rows["_score"].iloc[0])

    if np.isfinite(base_score):
        if cfg["higher_is_better"]:
            improvement = best_score - base_score
        else:
            improvement = (base_score - best_score) / abs(base_score) if base_score else np.nan
        threshold = cfg["min_improvement"]
        beats = bool(np.isfinite(improvement) and improvement >= threshold)
        if cfg["higher_is_better"]:
            detail = (f"{metric} {best_score:.3f} vs baseline {base_score:.3f} "
                      f"(+{improvement:.3f}; mínimo {threshold:.3f}).")
        else:
            detail = (f"{metric} {best_score:,.2f} vs baseline {base_score:,.2f} "
                      f"({improvement:.1%} mejor; mínimo {threshold:.0%}).")
    else:
        improvement = np.nan
        beats = True
        detail = f"No hay baseline comparable; se usa {metric} del conjunto de prueba."

    confiable = bool(beats)
    reason = (
        f"Seleccionado {best}: {detail}"
        if confiable
        else f"{best} no demuestra una mejora suficiente sobre el baseline. {detail} "
             "Trátese como modelo exploratorio, no como evidencia predictiva."
    )

    return {
        "best": best,
        "metric": metric,
        "score": best_score,
        "baseline_score": base_score,
        "improvement": float(improvement) if np.isfinite(improvement) else None,
        "beats_baseline": confiable,
        "confiable": confiable,
        "reason": reason,
        "ranking": ranked[["Modelo", "_score"]].rename(columns={"_score": metric})
        .reset_index(drop=True),
    }
