# Arquitectura del Sistema

## Flujo de ejecución
El sistema separa la interfaz y la ejecución por línea de comandos (`app.py` y `scripts/train.py`) de la lógica de procesamiento y modelado ubicada en `src/`.

## Flujo en `scripts/train.py`

El script acepta `reporte` (`.xlsx` o `.csv`) y, de forma opcional, `costos.csv` (`sku,costo_unitario`):

1. `raw = dp.load_file(sys.argv[1])` y `df, log = dp.clean(raw)` — carga del reporte y limpieza con registro de eventos (`log`).
2. `rep = dp.quality_report(df)` — cálculo de indicadores de calidad (imprimiendo el resumen sin `faltantes_pct` en consola).
3. `rm.diagnose(df)` — diagnóstico previo del objetivo de devoluciones.
4. `pm.train(df, costs)` — entrenamiento del modelo de rentabilidad; si `viable` es `True`, expone `tipo_resultado`, `n_train`, `n_test`, `metrics` e `importancia` (en `train.py` se muestran las 8 primeras filas de `importancia`); si no, devuelve `razon`.
5. `rm.train(df)` — entrenamiento del modelo de devoluciones; si `viable` es `True`, expone `n_train`, `n_test`, `tasa_train`, `tasa_test`, `metrics` e `importancia` (en `train.py` se muestran las 8 primeras filas de `importancia`); si no, devuelve `razon`.

## Módulos en `src/`

| Módulo | Funciones/rol observado | Responsabilidad |
| :--- | :--- | :--- |
| `data_processing` (`dp`) | `load_file`, `clean`, `quality_report` | Lectura de archivos de 2 cabeceras, normalización, depuración de PII, cruce de costos y reporte de calidad. |
| `features` | `[Completar con funciones/variables de src/features.py]` | Construcción de variables pre-venta sin *data leakage* y definición de etiquetas objetivo. |
| `profitability_model` (`pm`) | `train` | Entrenamiento del modelo de rentabilidad y cálculo de métricas/importancia. |
| `returns_model` (`rm`) | `diagnose`, `train` | Diagnóstico de viabilidad/etiquetas y entrenamiento del modelo de devoluciones. |
| `evaluation` | `[Completar con funciones de src/evaluation.py]` | Comparación de modelos (*baseline* vs entrenados) e importancia por permutación. |
| `recommendations` | `[Completar con funciones de src/recommendations.py]` | Generación de recomendaciones accionables por SKU y transportista. |
