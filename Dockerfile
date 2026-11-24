# SpamShield - Dockerfile multi-stage (deux cibles : `frontend` et `api`).
# La dernière cible (`api`) est celle construite par défaut.

# ---------------------------------------------------------------- frontend ---
FROM python:3.11-slim AS frontend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
# Seule dépendance du frontend (le HTTP utilise la bibliothèque standard).
RUN pip install --no-cache-dir "streamlit>=1.30,<2"
COPY streamlit_app.py .
ENV API_URL=http://api:8000
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# --------------------------------------------------------------------- api ---
FROM python:3.11-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Le code du backend ET le modèle (.joblib) sont copiés dans l'image :
# le modèle est donc disponible dès le démarrage du conteneur.
COPY . .
# Module ASGI à lancer, modifiable sans toucher au backend (ex. app.main:app).
ENV API_MODULE=main:app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn ${API_MODULE} --host 0.0.0.0 --port 8000"]
