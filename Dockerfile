# SpamShield - Dockerfile multi-stage (deux cibles : `frontend` et `api`).
# La dernière cible (`api`) est celle construite par défaut.

# ---------------------------------------------------------------- frontend ---
FROM python:3.13-slim AS frontend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
# Seule dépendance du frontend (le HTTP utilise la bibliothèque standard).
RUN pip install --no-cache-dir "streamlit==1.67.0"
COPY streamlit_app.py .
ENV API_URL=http://api:8000
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# --------------------------------------------------------------------- api ---
FROM python:3.13-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Copies ciblées : seulement le nécessaire pour l'API
COPY app/ ./app/
COPY artifacts/ ./artifacts/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]