"""Point d'entrée de l'API FastAPI SpamShield."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.services.predictor import load_model, predict

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, le stocke dans app.state.model."""
    model_path = load_model.__globals__["get_model_path"]()
    try:
        app.state.model = load_model(model_path)
        logger.info("Modèle chargé avec succès depuis %s", model_path)
    except FileNotFoundError as e:
        app.state.model = None
        logger.error("Modèle introuvable au démarrage : %s", e)
    yield
    # Nettoyage si nécessaire
    app.state.model = None


app = FastAPI(
    title="SpamShield API",
    description="API de détection de spam SMS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check - toujours disponible, même si le modèle est absent."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """Prédit si un SMS est spam ou légitime."""
    model = app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Vérifiez que le fichier artifacts/spam_pipeline.joblib existe "
            "et lancez l'entraînement avec : python src/model/train.py",
        )

    try:
        label = predict(model, request.text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(prediction=label)