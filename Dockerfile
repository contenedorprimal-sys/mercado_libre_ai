# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# ML Analytics IA — imagen de producción para Streamlit
# Base: Python 3.12 slim (Debian). Se instala libgomp1, requerido por XGBoost.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

LABEL org.opencontainers.image.title="ML Analytics IA" \
      org.opencontainers.image.description="Predicción de rentabilidad y riesgo de devoluciones (Mercado Libre)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# XGBoost necesita libgomp1 en tiempo de ejecución.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios con el mismo UID/GID habitual del host (1000)
# para que los volúmenes montados ./data y ./models sean escribibles.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

# Capa de dependencias (se cachea mientras requirements.txt no cambie).
COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación.
COPY --chown=app:app . .

# Directorios persistentes (en docker-compose se montan desde el host).
RUN mkdir -p /app/data /app/models \
    && chown -R app:app /app/data /app/models

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read().strip()==b'ok' else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
