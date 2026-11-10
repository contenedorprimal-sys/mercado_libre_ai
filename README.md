# Sistema Inteligente de Predicción de Rentabilidad y Riesgo de Devoluciones (Mercado Libre)

Interfaz Streamlit + modelos de machine learning para analizar ventas de Mercado Libre:
resultado económico por venta, riesgo de devoluciones, comparación de modelos y
recomendaciones por SKU/transportista. El procesamiento es local y elimina datos personales.

## Inicio rápido con Docker (recomendado)

```bash
docker compose up --build -d      # construye la imagen y arranca en segundo plano
docker compose ps                 # debe mostrar "healthy"
```

Abre: **http://localhost:8501**

La primera vez se carga automáticamente `data/demo_simulado.csv` para que la interfaz
tenga datos. También puedes subir tu propio reporte en *Carga de datos*.

```bash
docker compose logs -f app        # logs en vivo
docker compose down               # detener (los datos y modelos se conservan)
docker compose up -d              # volver a arrancar
```

### Persistencia

Los volúmenes están montados sobre el proyecto, así que **sobreviven a reinicios y
reconstrucciones**:

| Ruta del host | Ruta en el contenedor | Contenido |
|---|---|---|
| `./data` | `/app/data` | `data/current/` con el dataset y costos cargados + metadatos |
| `./models` | `/app/models` | `profitability.joblib` y `returns.joblib` entrenados |

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
- Prueba con datos simulados: `python scripts/make_demo_data.py` y luego
  `python scripts/train.py data/demo_simulado.csv`
  (las métricas de esos datos **NO** son resultados reales).

## Secciones de la interfaz

1. **Inicio** — indicadores clave, evolución mensual, top SKUs y actividad reciente.
2. **Carga de datos** — carga y limpieza, calidad, diagnóstico de devoluciones y vista previa.
3. **Análisis exploratorio** — productos, envíos, precios y resultado económico.
4. **Rentabilidad** — entrenamiento, métricas, importancia y real vs predicho.
5. **Predicción de devoluciones** — bandas de riesgo relativo y descarga por venta.
6. **Comparación de modelos** — métricas lado a lado (RMSE / ROC-AUC).
7. **Recomendaciones inteligentes** — hallazgos accionables por SKU y transportista.
8. **Reportes** — exportación de datos, métricas y resumen de la sesión.

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

## Estructura

```
app.py                  Interfaz Streamlit (UI + persistencia)
src/                    data_processing, features, profitability_model,
                        returns_model, evaluation, recommendations
scripts/                make_demo_data.py, train.py
data/                   demo_simulado.csv + current/ (sesión persistida)
models/                 modelos .joblib entrenados
Dockerfile              imagen de producción (usuario no root + healthcheck)
docker-compose.yml      servicio con volúmenes persistentes
.streamlit/config.toml  tema y configuración del servidor
```
