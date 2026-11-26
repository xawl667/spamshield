"""Modèles Pydantic pour l'API SpamShield."""

from typing import Literal

from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    """Requête de prédiction : un texte brut."""

    text: str

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Le champ 'text' doit être une chaîne de caractères")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Le texte ne doit pas être vide ou composé uniquement d'espaces")
        return stripped


class PredictResponse(BaseModel):
    """Réponse de prédiction."""

    prediction: Literal["spam", "legitime"]


class HealthResponse(BaseModel):
    """Réponse de health check."""

    status: str