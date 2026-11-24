# Motor de Recomendaciones y Operación

## Recomendaciones accionables

La pestaña **Recomendaciones inteligentes** de la interfaz presenta hallazgos accionables agrupados por **SKU** y por **transportista** (como el comportamiento de devoluciones asociado a transportistas como `DHL` o diferencias por producto).

## Entorno y despliegue

- **Contenedor:** construido desde `python:3.12-slim` con `libgomp1` (necesaria para XGBoost), etiqueta `mercado-libre-ai:latest`, usuario no `root` (`UID` / `GID` por defecto `1000`), `healthcheck` y puerto `8501` (`APP_PORT`).
- **Limpieza de estado persistido:** borrar `data/current` y `models/*.joblib` y ejecutar `docker compose restart` restablece la sesión (cargando `data/demo_simulado.csv` si `AUTO_LOAD_DEMO=1`).
