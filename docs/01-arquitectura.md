# Arquitectura del Sistema

## Visión general

El sistema combina una interfaz Streamlit (`app.py`), una ejecución por línea de
comandos (`scripts/train.py`) y un núcleo de análisis en `src/`. El núcleo **no
importa Streamlit** ni escribe en disco por sí mismo (salvo a través de
`save_model`/`joblib`), de modo que puede probarse sin interfaz.

## Capas (Clean / Hexagonal, aplicado de forma incremental)

| Capa | Archivos | Responsabilidad |
|---|---|---|
| **Dominio / reglas puras** | `selection.py`, `multivariate.py`, `evaluation.py`, `returns_model.build_target` | Criterio de selección, métricas, etiquetas, correlación/VIF/PCA. Sin UI ni persistencia. |
| **Casos de uso (ML)** | `profitability_model.py`, `returns_model.py`, `unsupervised_model.py`, `recommendations.py`, `features.py`, `data_processing.py` | Orquestan limpieza, features, entrenamiento, comparación y recomendaciones. |
| **Adaptadores de entrada** | `app.py` (UI), `scripts/train.py` (CLI), `scripts/make_demo_data.py` | Traducen interacción a llamadas del núcleo y presentan resultados. |
| **Infraestructura** | `models/*.joblib`, `data/current/`, volumen Docker | Persistencia de modelos, dataset y metadatos. Hoy parte vive en `app.py`. |

**Regla de dependencia:** las flechas apuntan hacia dentro. `src/` no conoce
`app.py`; la UI y el CLI dependen de `src/`, nunca al revés. Las reglas
(`selection.py`, `multivariate.py`) son funciones puras y se cubren con pruebas.

### Deuda arquitectónica consciente

`app.py` aún concentra persistencia (`save_dataset`, `load_saved`,
`hydrate_once`) y presentación. La extracción a `src/infrastructure/` es el
siguiente paso natural, pero se pospuso para no romper la app en esta iteración.
Del mismo modo, los módulos de casos de uso devuelven diccionarios en lugar de
DTOs tipados: es suficiente para el tamaño actual y mantiene compatibilidad.

## Flujo de ejecución en `scripts/train.py`

1. **Carga y limpieza:** `dp.load_file` + `dp.clean` → `(df, log)`.
2. **Calidad:** `dp.quality_report` (sin imprimir `faltantes_pct` en consola).
3. **Diagnóstico de devoluciones:** `rm.diagnose`.
4. **Rentabilidad:** `pm.train(df, costs)`; imprime métricas, mejor modelo,
   importancia y el criterio de selección devuelto en `seleccion`.
5. **Devoluciones:** `rm.train(df)` con el mismo criterio.
6. **No supervisado:** `um.fit_unsupervised(df)`; guarda
   `models/unsupervised.joblib`.
7. **Multivariado:** `mv.multivariate_report(df)` (vía del resultado anterior).

## Interfaz entre `scripts/train.py` y `src/`

| Componente | Llamada | Retorno |
|---|---|---|
| `data_processing` | `load_file`, `clean`, `quality_report` | `df`, `log`, métricas de calidad |
| `profitability_model` | `train(df, costs)` | `viable`, `metrics`, `mejor`, `importancia`, `seleccion`, `confiable` |
| `returns_model` | `diagnose(df)`, `train(df)` | diagnóstico y mismas claves de selección |
| `unsupervised_model` | `fit_unsupervised(df)`, `save_model`, `predict_unsupervised` | clústeres, anomalías, PCA, multivariado |
| `multivariate` | `multivariate_report(df)` | correlación, VIF, PCA y aplicabilidad |
| `selection` | `choose_model(metrics, task)` | `best`, `confiable`, `reason`, `ranking` |
