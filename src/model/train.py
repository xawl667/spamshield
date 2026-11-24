"""
SpamShield - entraînement du modèle de détection de spam.

Lancement (depuis la racine du projet) :
    python src/model/train.py

Entrée  : data/processed/sms_clean.csv  (colonnes : text, label)
Sorties : artifacts/spam_pipeline.joblib
          reports/metrics.json
          reports/misclassified_examples.json
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Chemins calculés depuis l'emplacement du fichier : le script marche
# quel que soit le dossier depuis lequel on le lance.
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "sms_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
PIPELINE_PATH = ARTIFACTS_DIR / "spam_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
ERRORS_PATH = REPORTS_DIR / "misclassified_examples.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_LABEL = "spam"
ALLOWED_LABELS = {"spam", "legitime"}
N_ERRORS_PER_TYPE = 5

# Hyperparamètres fixés a priori (PAS réglés sur le jeu de test).
TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
NB_PARAMS = {"alpha": 0.1}


def load_dataset(path: Path) -> pd.DataFrame:
    """Charge le CSV et vérifie le contrat d'entrée."""
    if not path.exists():
        sys.exit(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    missing = {"text", "label"} - set(df.columns)
    if missing:
        sys.exit(f"Colonnes manquantes : {sorted(missing)}")

    if df[["text", "label"]].isna().any().any():
        sys.exit("Valeurs manquantes dans text/label : le nettoyage est incomplet.")

    unknown = set(df["label"].unique()) - ALLOWED_LABELS
    if unknown:
        sys.exit(f"Labels non autorisés : {sorted(unknown)}")

    return df[["text", "label"]]


def compute_metrics(y_true, y_pred) -> dict:
    """Métriques centrées sur la classe spam (+ accuracy en information)."""
    labels = ["legitime", "spam"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        "precision_spam": precision_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "recall_spam": recall_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "f1_spam": f1_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": {
            "vrais_negatifs_legitime_ok": int(tn),
            "faux_positifs_legitime_classe_spam": int(fp),
            "faux_negatifs_spam_non_detecte": int(fn),
            "vrais_positifs_spam_detecte": int(tp),
        },
    }


def collect_errors(X_test, y_test, y_pred) -> dict:
    """Quelques exemples de faux positifs et de faux négatifs."""
    res = pd.DataFrame(
        {"text": X_test.values, "reel": y_test.values, "predit": y_pred}
    )
    fp = res[(res["reel"] == "legitime") & (res["predit"] == "spam")]
    fn = res[(res["reel"] == "spam") & (res["predit"] == "legitime")]
    return {
        "faux_positifs_legitime_classe_spam": fp.head(N_ERRORS_PER_TYPE).to_dict("records"),
        "faux_negatifs_spam_non_detecte": fn.head(N_ERRORS_PER_TYPE).to_dict("records"),
        "total_faux_positifs": int(len(fp)),
        "total_faux_negatifs": int(len(fn)),
    }


def main() -> None:
    # 1. Chargement
    df = load_dataset(DATA_PATH)
    print(f"Dataset chargé : {len(df)} SMS")
    print(df["label"].value_counts().to_string(), "\n")

    # 2. Séparation stratifiée 80/20 : un seul split pour les deux modèles
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    print(f"Train : {len(X_train)} | Test : {len(X_test)}\n")

    # 3. Baseline : prédit toujours la classe majoritaire du train
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)

    # 4. Pipeline TF-IDF + Naive Bayes, entraîné uniquement sur le train
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", MultinomialNB(**NB_PARAMS)),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred_model = pipeline.predict(X_test)

    # 5. Comparaison sur le même jeu de test
    metrics_base = compute_metrics(y_test, y_pred_base)
    metrics_model = compute_metrics(y_test, y_pred_model)

    print("=== Résultats sur le jeu de test (classe spam) ===")
    print(f"{'':22s}{'Baseline':>10s}{'TF-IDF+NB':>12s}")
    for key in ["precision_spam", "recall_spam", "f1_spam", "accuracy"]:
        print(f"{key:22s}{metrics_base[key]:10.4f}{metrics_model[key]:12.4f}")
    print()

    # 6. Erreurs de classification
    errors = collect_errors(X_test, y_test, y_pred_model)
    print(
        f"Erreurs du modèle : {errors['total_faux_positifs']} faux positifs, "
        f"{errors['total_faux_negatifs']} faux négatifs"
    )
    for kind in ["faux_positifs_legitime_classe_spam", "faux_negatifs_spam_non_detecte"]:
        print(f"\n--- {kind} (exemples) ---")
        for row in errors[kind]:
            print(f"  - {row['text'][:110]}")

    # 7. Export du pipeline complet + rapports
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PIPELINE_PATH)

    report = {
        "dataset": {
            "path": str(DATA_PATH.relative_to(ROOT)),
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "repartition_totale": df["label"].value_counts().to_dict(),
            "repartition_test": y_test.value_counts().to_dict(),
        },
        "split": {
            "test_size": TEST_SIZE,
            "stratify": "label",
            "random_state": RANDOM_STATE,
        },
        "parametres_modele": {
            "tfidf": {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
            "multinomial_nb": NB_PARAMS,
            "baseline": {"strategy": "most_frequent"},
            "note": "Hyperparamètres fixés a priori, non réglés sur le test.",
        },
        "classes_pipeline": [str(c) for c in pipeline.classes_],
        "taille_vocabulaire_tfidf": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "metriques": {"baseline": metrics_base, "tfidf_multinomial_nb": metrics_model},
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__},
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ERRORS_PATH.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    # 8. Test après export : on recharge le fichier et on vérifie le contrat
    reloaded = joblib.load(PIPELINE_PATH)
    raw_texts = [
        "WINNER!! You have won a free prize, call now to claim",
        "Ok, see you at 7pm at the station",
    ]
    preds = reloaded.predict(raw_texts)
    assert set(preds) <= ALLOWED_LABELS, f"Labels inattendus : {set(preds)}"
    assert (reloaded.predict(X_test) == y_pred_model).all(), "Prédictions différentes après rechargement"
    print("\nTest joblib.load() OK ->", dict(zip(raw_texts, [str(p) for p in preds])))
    print(f"Pipeline : {PIPELINE_PATH.relative_to(ROOT)}")
    print(f"Métriques : {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()