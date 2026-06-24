FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY sql/ sql/
COPY data/processed/ data/processed/

# ─── API target ──────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Dashboard target ────────────────────────────────────────
FROM base AS dashboard
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
