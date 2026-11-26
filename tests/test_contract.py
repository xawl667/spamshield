"""Tests de contrat : dataset et pipeline."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.services.predictor import DEFAULT_MODEL_PATH, load_model, predict


class TestDatasetContract:
    """Vérifie que le dataset nettoyé respecte le contrat."""

    CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "sms_clean.csv"

    def test_csv_exists(self):
        assert self.CSV_PATH.exists(), f"Fichier introuvable : {self.CSV_PATH}"

    def test_columns_exact(self):
        df = pd.read_csv(self.CSV_PATH)
        assert list(df.columns) == ["text", "label"], f"Colonnes inattendues : {list(df.columns)}"

    def test_labels_allowed(self):
        df = pd.read_csv(self.CSV_PATH)
        allowed = {"spam", "legitime"}
        assert set(df["label"].unique()) == allowed, f"Labels invalides : {df['label'].unique()}"

    def test_no_empty_text(self):
        df = pd.read_csv(self.CSV_PATH)
        assert (df["text"].str.strip() == "").sum() == 0, "Textes vides détectés"

    def test_no_null_values(self):
        df = pd.read_csv(self.CSV_PATH)
        assert df[["text", "label"]].isna().sum().sum() == 0, "Valeurs manquantes détectées"


class TestPipelineContract:
    """Vérifie que le pipeline se charge et prédit correctement."""

    def test_pipeline_loads(self):
        pipeline = load_model(DEFAULT_MODEL_PATH)
        assert pipeline is not None

    def test_pipeline_predicts_allowed_labels(self):
        pipeline = load_model(DEFAULT_MODEL_PATH)
        test_texts = [
            "WINNER!! Call now to claim your prize!",
            "Hey, are we meeting tomorrow?",
        ]
        for text in test_texts:
            pred = predict(pipeline, text)
            assert pred in {"spam", "legitime"}, f"Label inattendu : {pred} pour '{text}'"