# Sistema Inteligente de Predicción de Rentabilidad y Riesgo de Devoluciones (Mercado Libre)

Interfaz Streamlit + modelos de machine learning para analizar ventas de Mercado Libre:
resultado económico por venta, riesgo de devoluciones, **segmentación no supervisada**,
comparación de modelos y recomendaciones por SKU/transportista. El procesamiento es
local y elimina datos personales.

## Inicio rápido con Docker (recomendado)

```bash
docker compose up --build -d      # construye la imagen y arranca en segundo plano
docker compose ps                 # debe mostrar "healthy"
```
docker compose logs -f app        # logs en vivo
docker compose down               # detener (los datos y modelos se conservan)
docker compose up -d              # volver a arrancar

### Persistencia

Los volúmenes están montados sobre el proyecto, así que **sobreviven a reinicios y
reconstrucciones**:

| Ruta del host | Ruta en el contenedor | Contenido |
|---|---|---|
| `./data` | `/app/data` | `data/current/` con el dataset y costos cargados + metadatos |
| `./models` | `/app/models` | `profitability.joblib`, `returns.joblib` y `unsupervised.joblib` |

Al abrir la app se recuperan automáticamente la última sesión de datos y los modelos
guardados (aparecen como *guardado* en la barra lateral). Para reiniciar de cero:

```bash
rm -rf data/current models/*.joblib
docker compose restart
```

### Variables de entorno (docker-compose.yml)

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `APP_PORT` | `8501` | Puerto publicado en el host |
| `AUTO_LOAD_DEMO` | `1` | Carga el CSV simulado si no hay sesión guardada |
| `UID` / `GID` | `1000` | Usuario del host dueño de los volúmenes |

Ejemplo con otro puerto: `APP_PORT=8080 docker compose up -d`.

### Imágenes Docker

La imagen se construye a partir de `python:3.12-slim` e instala `libgomp1`
(requerido por XGBoost). Resultado: **`mercado-libre-ai:latest`**.

```bash
docker images | grep -E 'mercado|python'
```

Si ya existía una imagen `mercado_ai:latest` de una versión anterior del proyecto,
no se reutiliza: el `docker-compose.yml` construye y etiqueta `mercado-libre-ai:latest`.

## Uso local sin Docker

```bash

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

- Interfaz: sube tu Excel/CSV en *Carga de datos* (opcionalmente `costos.csv` con `sku,costo_unitario`).
- Terminal: `python scripts/train.py data/mi_reporte.xlsx [data/costos.csv]`
- Pruebas: `python -m unittest discover -s tests -v`
- Prueba con datos simulados: `python scripts/make_demo_data.py` y luego
  `python scripts/train.py data/demo_simulado.csv`
  (las métricas de esos datos **NO** son resultados reales).

  ### Qué hace `scripts/train.py` al ejecutarse

1. Carga el reporte con `data_processing.load_file()` y lo limpia con `data_processing.clean()`, imprimiendo columnas finales y el registro (`log`) de limpieza.
2. Muestra el resumen de calidad (`quality_report`, sin `faltantes_pct` en consola) y el diagnóstico de devoluciones (`returns_model.diagnose()`).
3. Ejecuta `profitability_model.train(df, costs)` y `returns_model.train(df)`, imprimiendo métricas, el mejor modelo, el **criterio de selección** (¿supera al baseline?) y el top 8 de importancia, o el motivo (`razon`) cuando un modelo no es viable.
4. Ejecuta `unsupervised_model.fit_unsupervised(df)`: segmentación KMeans, anomalías y PCA; guarda `models/unsupervised.joblib`.
5. Imprime el análisis multivariado (`multivariate.multivariate_report`): correlación, VIF, varianza explicada y aplicabilidad.

### Formato esperado de archivos de entrada

- **Reporte de ventas (`.xlsx` / `.csv`):** exportación estándar de ventas de Mercado Libre. Debe contener al menos identificador de venta, fecha, SKU/publicación, unidades, precio unitario, tipo de publicación, método de envío, estado de entrega/reclamo y la columna `Total (MXN)`.
- **Catálogo de costos (`costos.csv`, opcional):** archivo UTF-8 separado por comas con encabezado exacto `sku,costo_unitario` (valores numéricos en MXN sin símbolo `$`).

## Secciones de la interfaz

1. **Inicio** — indicadores clave, evolución mensual, top SKUs y actividad reciente.
2. **Carga de datos** — carga y limpieza, calidad, diagnóstico de devoluciones y vista previa.
3. **Análisis exploratorio** — productos, envíos, precios y resultado económico.
4. **Rentabilidad** — entrenamiento, métricas, importancia y real vs predicho.
5. **Predicción de devoluciones** — bandas de riesgo relativo y descarga por venta.
6. **Comparación de modelos** — métricas lado a lado (RMSE / ROC-AUC) y criterio de selección.
7. **Segmentación y anomalías** — KMeans, IsolationForest, mapa PCA y análisis multivariado (correlación/VIF).
8. **Recomendaciones inteligentes** — hallazgos accionables por SKU y transportista.
9. **Reportes** — exportación de datos, métricas, segmentos y resumen de la sesión.

## Decisiones metodológicas

- **Rentabilidad:** resultado = `Total (MXN)` de Mercado Libre − costo de adquisición.
  `Total` ya incluye cargos, envíos, descuentos y reembolsos (sin doble conteo).
  Sin catálogo de costos se etiqueta como **PARCIAL**.
- **Sin fuga de información:** los predictores solo usan datos previos a la venta
  (precio, unidades, SKU, tipo de publicación, envío, mes). No se usan ingresos,
  cargos, costos ni reembolsos reales.
- **Validación temporal:** 80 % más antiguo = entrenamiento, 20 % más reciente = prueba.
  Se incluye un *baseline* para comparar.
- **Devoluciones:** 1 = evidencia de devolución; 0 = entregada hace ≥30 días sin reclamo
  ni devolución; el resto es ambiguo y se excluye. Si hay menos de 30 positivos, el
  modelo NO se entrena y se explica por qué.
- **Privacidad:** columnas personales (comprador, domicilio, documentos, seguimiento, etc.)
  se eliminan al cargar.
- **Importancia de variables:** permutación (asociación, no causalidad).
- **Selección de modelos:** regla única en `src/selection.py`. Regresión → menor RMSE;
  clasificación → mayor ROC-AUC; en ambos casos debe superar al baseline (≥1 % y
  ≥0.01 respectivamente) o se marca como no confiable.
- **No supervisado:** KMeans elige `k` por silueta; IsolationForest marca el 5 % más
  atípico. El resultado económico se usa solo para interpretar los segmentos.
- **Multivariado:** correlación, VIF y PCA exploratorios. No se usa MANOVA porque los
  datos son un registro completo, no una muestra aleatoria.

### Resultado con datos simulados

| Tarea | Mejor modelo | Métrica | Baseline |
|---|---|---|---|
| Rentabilidad | Random Forest | RMSE 605.39 · R² 0.438 | RMSE 814.28 |
| Devoluciones | Regresión logística | ROC-AUC 0.614 | 0.500 |
| Segmentación | KMeans k=2 | Silueta 0.206 (débil) | — |

Ver [auditoría de modelos](docs/05-auditoria-modelos.md) para hallazgos y limitaciones.

## Documentación técnica

Para detalles de implementación, contratos de módulos y especificaciones de modelos, consulta:
- [Documento de negocio, alcance y limitaciones](docs/00-documento-negocio.md) — propósito, reglas operativas, lo que el sistema hace y lo que deja de hacer.
- [Arquitectura y flujo del sistema](docs/01-arquitectura.md) — diagrama de componentes y responsabilidades de `src/`.
- [Diccionario de datos y privacidad](docs/02-datos-y-privacidad.md) — esquema esperado, limpieza, anonimización y persistencia.
- [Especificación de modelos ML](docs/03-modelos-ml.md) — ingeniería de variables, algoritmos, validación temporal y bandas de riesgo.
- [Motor de recomendaciones y operación](docs/04-recomendaciones-y-operacion.md) — lógica de hallazgos por SKU/transportista y troubleshooting.
- [Auditoría de modelos y análisis multivariado](docs/05-auditoria-modelos.md) — hallazgos, criterio de selección, mejor modelo y limitaciones.

## Estructura

```
app.py                  Interfaz Streamlit (UI + persistencia)
src/                    data_processing, features, evaluation,
                        profitability_model, returns_model,
                        unsupervised_model, multivariate, selection,
                        recommendations
scripts/                make_demo_data.py, train.py
tests/                  unittest del criterio de selección, multivariado
                        y modelo no supervisado
data/                   demo_simulado.csv + current/ (sesión persistida)
models/                 profitability.joblib, returns.joblib, unsupervised.joblib
Dockerfile              imagen de producción (usuario no root + healthcheck)
docker-compose.yml      servicio con volúmenes persistentes
.streamlit/config.toml  tema y configuración del servidor
```
