# Especificación de Modelos de Machine Learning

## Reglas metodológicas

- **Sin fuga de información:** los predictores no incluyen ingresos, cargos,
  costos reales, reembolsos ni el resultado de la devolución. Los clústeres se
  forman sin el resultado económico (solo se usa para interpretarlos).
- **Validación temporal:** el 80 % más antiguo entrena y el 20 % más reciente
  prueba. No se mezcla al azar porque rompería el orden temporal.
- **Selección única:** `src/selection.py` elige al mejor modelo con una sola
  regla (RMSE para regresión, ROC-AUC para clasificación) y exige superar al
  baseline. Si no lo supera, se marca `confiable = False`.
- **Importancia de variables:** permutación sobre variables originales; refleja
  asociación, no causalidad.

## Modelo de rentabilidad (`profitability_model.py`)

- **Objetivo:** `Total (MXN)` de Mercado Libre menos el costo de adquisición;
  sin catálogo de costos se etiqueta como **PARCIAL**.
- **Comparación:** baseline (media), regresión lineal, Random Forest y XGBoost
  (si está instalado).
- **Salida:** diccionario con `viable`, `tipo_resultado`, `n_train`, `n_test`,
  `metrics`, `mejor`, `importancia`, `seleccion`, `confiable`, `advertencia` y
  `errores` (modelos que no pudieron entrenarse).

## Modelo de devoluciones (`returns_model.py`)

- **Etiquetado:** `1` = evidencia de devolución; `0` = entregada hace ≥30 días
  sin reclamo ni devolución; casos ambiguos excluidos.
- **Condición mínima:** ≥30 positivos y ≥100 negativos para entrenar; el periodo
  de prueba debe tener ≥5 positivos. Si no, `viable = False` y se explica.
- **Bandas de riesgo:** percentiles 50 y 80 de la probabilidad estimada. Las
  probabilidades **no están calibradas** con `class_weight`; por eso se habla de
  riesgo relativo.

## Aprendizaje no supervisado (`unsupervised_model.py`)

- **KMeans:** agrupa ventas con variables pre-venta. `k` se elige por silueta
  (2–8) y se reporta la inercia para el método del codo.
- **IsolationForest:** anomalías con `contamination=0.05` (5 %). Se descartó
  `"auto"` porque marcaba más de la mitad de las ventas.
- **PCA (2 componentes):** visualiza la separación de los segmentos.
- **Perfil de segmento:** medias de variables de entrada + resultado económico y
  tasa de devolución observados (solo interpretación).

## Análisis multivariado (`multivariate.py`)

- **Correlación** y **VIF** (desde la inversa de la matriz de correlación) para
  detectar redundancia; se descartan columnas constantes o vacías.
- **PCA** sobre variables estandarizadas: varianza explicada, componentes
  necesarios para 80 % y cargas.
- Es **exploratorio**. No se usa MANOVA/Hotelling: el registro es censal, no una
  muestra aleatoria, y hay colinealidad severa.

## Estado actual con datos simulados

| Tarea | Mejor modelo | Métrica | Baseline | ¿Confiable? |
|---|---|---|---|---|
| Rentabilidad | Random Forest | RMSE 605.39 / R² 0.438 | RMSE 814.28 | Sí |
| Devoluciones | Regresión logística | ROC-AUC 0.614 | 0.500 | Sí, pero débil |
| Segmentación | KMeans k=2 | Silueta 0.206 | — | Estructura débil |

Detalle y limitaciones en [`05-auditoria-modelos.md`](05-auditoria-modelos.md).
