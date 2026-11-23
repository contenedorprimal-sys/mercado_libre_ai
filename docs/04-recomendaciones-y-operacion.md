# Motor de Recomendaciones y Operación

## Lógica de recomendaciones (`recommendations.py`)
- **Por SKU:** identifica productos con alta probabilidad predicha de devolución o margen negativo recurrente.
- **Por transportista / logística:** detecta desviaciones significativas en tasa de devolución frente al promedio global del catálogo.

## Operación en contenedor (`Dockerfile` / `docker-compose.yml`)
- **Usuario:** ejecución sin privilegios `root` (`UID`/`GID` configurables).
- **Healthcheck:** verifica disponibilidad del servidor Streamlit en el puerto `8501` antes de marcar el contenedor como `healthy`.
- **Dependencia crítica:** `libgomp1` instalada en `python:3.12-slim` para soporte OpenMP de XGBoost.
