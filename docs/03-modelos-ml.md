# Especificación de Modelos de Machine Learning

## Estrategia de validación
- Ordenamiento cronológico por fecha de venta.
- **Train (80 % más antiguo) / Test (20 % más reciente):** evita fuga temporal (*look-ahead bias*).

## Variables predictoras (`features.py`)
Solo se incluyen atributos conocidos **antes** de concretar la venta:
- Precio publicado, cantidad de unidades, SKU / categoría, tipo de publicación, modalidad/transportista de envío y estacionalidad (mes / día de semana).
- **Excluidas explícitamente:** comisiones reales cobradas, costo de envío final liquidado, reembolsos y costos de adquisición.

## Modelo 1: Rentabilidad (`profitability.joblib`)
- **Objetivo:** margen económico neto (`Total (MXN) - costo_unitario * unidades`).
- **Métrica principal:** RMSE (y comparación contra *baseline* de media/mediana histórica).

## Modelo 2: Riesgo de devoluciones (`returns.joblib`)
- **Condición de activación:** mínimo 30 casos positivos en el conjunto de entrenamiento.
- **Métrica principal:** ROC-AUC.
- **Bandas de riesgo:** segmentación por cuantiles/umbrales de probabilidad predicha (Bajo, Medio, Alto) para descarga operativa por venta.
