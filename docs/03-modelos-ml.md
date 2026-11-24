# Especificación de Modelos de Machine Learning

## Reglas generales

- **Sin fuga de información (*no leakage*):** solo variables conocidas antes de concretar la venta (precio, unidades, SKU, tipo de publicación, envío, mes); se excluyen ingresos finales, cargos, reembolsos y costos reales como predictores.
- **Validación temporal:** 80 % más antiguo para entrenamiento y 20 % más reciente para prueba, comparando contra *baseline*.
- **Importancia de variables:** calculada mediante permutación (mide asociación predictiva, no causalidad).

## Modelo de rentabilidad (`profitability_model.py`)

- **Objetivo:** `Total (MXN)` − costo de adquisición (etiquetado como **PARCIAL** si no se proporciona catálogo de costos).
- **Salida de `pm.train(df, costs)`:** diccionario con `viable`, y cuando es viable: `tipo_resultado`, `n_train`, `n_test`, `metrics` e `importancia`; en caso contrario, `razon`.
- `[Completar con algoritmo, hiperparámetros exactos y métricas reportadas en profitability_model.py / evaluation.py]`.

## Modelo de devoluciones (`returns_model.py`)

- **Definición de etiqueta:** `1` con evidencia de devolución; `0` si fue entregada hace `>= 30` días sin reclamo ni devolución; casos ambiguos excluidos.
- **Umbral mínimo:** requiere al menos `30` casos positivos para entrenarse; si no se cumple, `viable` es `False` y se explica en `razon`.
- **Salida de `rm.train(df)`:** diccionario con `viable`, y cuando es viable: `n_train`, `n_test`, `tasa_train`, `tasa_test`, `metrics` e `importancia`; en caso contrario, `razon`.
- `[Completar con algoritmo, hiperparámetros exactos y definición de bandas de riesgo desde returns_model.py / evaluation.py]`.
