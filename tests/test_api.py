"""Tests de l'API SpamShield avec le vrai modèle."""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


SPAM_TEXT = (
    "WINNER!! As a valued network customer you have been selected to receivea "
    "900 pound prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
)
LEGIT_TEXT = "Are we still meeting for lunch tomorrow?"


class TestHealth:
    """Tests du endpoint /health."""

    def test_health_ok(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestPredict:
    """Tests du endpoint /predict avec le vrai modèle."""

    def test_predict_spam(self):
        with TestClient(app) as client:
            resp = client.post("/predict", json={"text": SPAM_TEXT})
            assert resp.status_code == 200
            assert resp.json() == {"prediction": "spam"}

    def test_predict_legitime(self):
        with TestClient(app) as client:
            resp = client.post("/predict", json={"text": LEGIT_TEXT})
            assert resp.status_code == 200
            assert resp.json() == {"prediction": "legitime"}

    def test_predict_empty_text(self):
        with TestClient(app) as client:
            resp = client.post("/predict", json={"text": ""})
            assert resp.status_code == 422

    def test_predict_whitespace_only(self):
        with TestClient(app) as client:
            resp = client.post("/predict", json={"text": "   "})
            assert resp.status_code == 422

    def test_predict_missing_field(self):
        with TestClient(app) as client:
            resp = client.post("/predict", json={})
            assert resp.status_code == 422

    def test_predict_wrong_type(self):
        with TestClient(app) as client:
            resp = client.post("/predict", json={"text": 123})
            assert resp.status_code == 422


class TestModelMissing:
    """Tests quand le modèle est absent."""

    def test_predict_model_missing_returns_503(self, monkeypatch):
        # Créer un chemin inexistant
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "nonexistent.joblib"
            monkeypatch.setenv("MODEL_PATH", str(fake_path))

            # Important : recréer le client APRÈS le monkeypatch pour déclencher le lifespan
            with TestClient(app) as client:
                resp = client.post("/predict", json={"text": "test"})
                assert resp.status_code == 503
                assert "Modèle non chargé" in resp.json()["detail"]
                assert "spam_pipeline.joblib" in resp.json()["detail"]