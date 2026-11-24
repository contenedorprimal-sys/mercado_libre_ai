# Especificación de Modelos de Machine Learning

## Reglas metodológicas

- **Sin fuga de información:** los predictores emplean únicamente información previa a la venta (precio, unidades, SKU, tipo de publicación, envío, mes) y excluyen ingresos, cargos, costos o reembolsos reales.
- **Validación temporal:** división cronológica con 80 % más antiguo para entrenamiento y 20 % más reciente para prueba, contrastada contra un *baseline*.
- **Importancia de variables:** calculada por permutación (refleja asociación, no causalidad).

## Modelo de rentabilidad (`profitability_model.py`)

- **Definición del objetivo:** `Total (MXN)` de Mercado Libre menos el costo de adquisición; sin catálogo de costos (`costos.csv` con `sku,costo_unitario`) el resultado se etiqueta como **PARCIAL**.
- **Salida de `pm.train(df, costs)`:** diccionario con `viable` (`bool`). Si es viable, incluye `tipo_resultado`, `n_train`, `n_test`, `metrics` e `importancia`; si no lo es, incluye `razon`.

## Modelo de devoluciones (`returns_model.py`)

- **Etiquetado:** `1` = evidencia de devolución; `0` = entregada hace ≥30 días sin reclamo ni devolución; casos intermedios/ambiguos excluidos.
- **Condición mínima:** si hay menos de 30 casos positivos, el modelo no se entrena (`viable = False`) y se explica el motivo en `razon`.
- **Salida de `rm.train(df)`:** diccionario con `viable` (`bool`). Si es viable, incluye `n_train`, `n_test`, `tasa_train`, `tasa_test`, `metrics` e `importancia`; si no lo es, incluye `razon`.
