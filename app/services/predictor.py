"""Chargement du modèle et fonction de prédiction."""

import os
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

ALLOWED_LABELS = {"spam", "legitime"}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "spam_pipeline.joblib"


def get_model_path() -> Path:
    """Retourne le chemin du modèle, surchargeable par la variable d'environnement MODEL_PATH."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_MODEL_PATH


def load_model(path: Path | None = None) -> Pipeline:
    """Charge le pipeline scikit-learn depuis le fichier joblib.

    Args:
        path: Chemin vers le fichier .joblib. Si None, utilise get_model_path().

    Returns:
        Le pipeline chargé.

    Raises:
        FileNotFoundError: Si le fichier modèle est introuvable, avec un message explicite.
    """
    model_path = path or get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. "
            "Lancez l'entraînement avec : python src/model/train.py"
        )
    return joblib.load(model_path)


def predict(pipeline: Pipeline, text: str) -> str:
    """Effectue une prédiction sur un texte brut.

    Args:
        pipeline: Pipeline scikit-learn chargé (TfidfVectorizer + MultinomialNB).
        text: Texte brut à classifier.

    Returns:
        Label prédit : "spam" ou "legitime".

    Raises:
        ValueError: Si la prédiction retourne un label inattendu.
    """
    pred = pipeline.predict([text])[0]
    pred_str = str(pred)
    if pred_str not in ALLOWED_LABELS:
        raise ValueError(f"Label inattendu du modèle : {pred_str!r} (attendu : spam ou legitime)")
    return pred_str