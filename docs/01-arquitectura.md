# Arquitectura del Sistema

## Flujo de ejecución

El sistema opera bajo dos puntos de entrada que comparten el mismo núcleo en `src/`:

1. **Interfaz interactiva (`app.py`):** gestiona estado de sesión en Streamlit, carga de archivos, persistencia en `data/current/` y visualización de resultados.
2. **Entrenamiento por CLI (`scripts/train.py`):** ejecuta el pipeline de forma desatendida y serializa artefactos en `models/`.

## Responsabilidad de módulos (`src/`)

| Módulo | Entrada | Salida | Responsabilidad principal |
| :--- | :--- | :--- | :--- |
| `data_processing.py` | DataFrame crudo + `costos.csv` | DataFrame limpio + metadatos | Normalización de columnas, cruce de costos, cálculo de `Total - costo` y purga de PII. |
| `features.py` | DataFrame limpio | Matriz $X$ (pre-venta) + targets $y$ | Construcción de variables sin *data leakage* y etiquetado de devoluciones ($\ge 30$ días). |
| `profitability_model.py` | $(X, y_{\text{rentabilidad}})$ | `profitability.joblib` + métricas | Entrenamiento de regresor (XGBoost / baseline) con split temporal 80/20. |
| `returns_model.py` | $(X, y_{\text{devolucion}})$ | `returns.joblib` + probabilidades | Clasificador binario (solo si positivos $\ge 30$) y asignación de bandas de riesgo relativo. |
| `evaluation.py` | Modelos + split de prueba | RMSE, MAE, ROC-AUC, importancia | Cálculo de métricas comparativas e importancia por permutación. |
| `recommendations.py` | Predicciones + agregados | Lista de hallazgos accionables | Reglas heurísticas/estadísticas por SKU y transportista. |
