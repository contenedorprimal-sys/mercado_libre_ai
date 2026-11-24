# Arquitectura del Sistema

## Visión general

El sistema combina la interfaz web en Streamlit (`app.py`) y la ejecución por línea de comandos (`scripts/train.py`), apoyándose en módulos de `src/` (`data_processing`, `profitability_model`, `returns_model`, además de `features`, `evaluation` y `recommendations`).

## Flujo de ejecución en `scripts/train.py`

El script recibe como argumento obligatorio la ruta del reporte (`.xlsx` o `.csv`) y, de forma opcional, la ruta a `costos.csv` (columnas `sku` y `costo_unitario`):

1. **Carga y limpieza:** `raw = dp.load_file(sys.argv[1])` y `df, log = dp.clean(raw)` — lee el archivo de entrada, aplica limpieza y devuelve el `DataFrame` depurado junto con la bitácora (`log`).
2. **Reporte de calidad:** `rep = dp.quality_report(df)` — genera indicadores de calidad e imprime el diccionario excluyendo la clave `faltantes_pct`.
3. **Diagnóstico de devoluciones:** `rm.diagnose(df)` — imprime el diagnóstico previo del objetivo de devoluciones.
4. **Entrenamiento de rentabilidad:** `p = pm.train(df, costs)` — si `p["viable"]` es verdadero, imprime `tipo_resultado`, `n_train`, `n_test`, la tabla `metrics` (redondeada a 3 decimales) y las primeras 8 filas de `importancia`; si no, imprime `p["razon"]`.
5. **Entrenamiento de devoluciones:** `r = rm.train(df)` — si `r["viable"]` es verdadero, imprime `n_train`, `n_test`, `tasa_train`, `tasa_test`, la tabla `metrics` (redondeada a 3 decimales) y las primeras 8 filas de `importancia`; si no, imprime `r["razon"]`.

## Interfaces observadas entre `scripts/train.py` y `src/`

| Componente | Llamada observada | Uso / retorno esperado |
| :--- | :--- | :--- |
| `data_processing` (`dp`) | `load_file(path)`, `clean(raw)`, `quality_report(df)` | Carga el reporte, devuelve `(df, log)` y genera métricas de calidad (incluyendo `faltantes_pct`). |
| `profitability_model` (`pm`) | `train(df, costs)` | Retorna diccionario con `viable` y, cuando es viable, `tipo_resultado`, `n_train`, `n_test`, `metrics`, `importancia`; en caso contrario, `razon`. |
| `returns_model` (`rm`) | `diagnose(df)`, `train(df)` | Diagnostica el objetivo y retorna diccionario con `viable` y, cuando es viable, `n_train`, `n_test`, `tasa_train`, `tasa_test`, `metrics`, `importancia`; en caso contrario, `razon`. |
