# SpamShield

SpamShield est une API de détection de spam SMS (spam / légitime) avec interface Streamlit. Développée pour le hackathon Data & IA HETIC.

```
Data (UCI) -> prepare.py -> sms_clean.csv
                    -> train.py -> artifacts/spam_pipeline.joblib
                                          -> FastAPI (port 8000) <- Docker Compose -> Streamlit (port 8501)
```

## Dataset

UCI SMS Spam Collection. Après nettoyage : 5 160 lignes, 12,4 % spam. Détails dans `data/DATA_CARD.md`.

## Modèle et résultats

| Modèle              | Precision (spam) | Recall (spam) | F1 (spam) | Accuracy |
|---------------------|------------------|---------------|-----------|----------|
| Baseline (majorité) | 0.00             | 0.00          | 0.00      | 0.876    |
| TF-IDF + NB         | 0.94             | 0.90          | 0.92      | 0.97     |

Matrice de confusion (TF-IDF+NB) : 3 faux positifs, 13 faux négatifs.

## Limites

- Rappel spam 0,90 : 10 % des spams passent au travers.
- Dataset anglais des années 2010, pas représentatif des SMS actuels (français, raccourcis, émojis).
- Hyperparamètres fixés a priori (pas de recherche), seuil de décision à 0,5.

## Lancement rapide

```bash
docker compose up --build -d
```

- API (Swagger) : http://localhost:8000/docs
- Frontend : http://localhost:8501

### Local (sans Docker)

```bash
pip install -r requirements.txt
python src/model/train.py
uvicorn app.main:app --reload
streamlit run streamlit_app.py
```

## Exemple d'appel API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER!! Claim your 900 pound prize now!"}'
```

Réponse :
```json
{"prediction": "spam"}
```

## Tests

```bash
pytest -q
# 15 tests passed
```

## Structure du repo

```
spamshield/
├── app/                 # FastAPI (main, schemas, services/predictor)
├── artifacts/           # spam_pipeline.joblib (modèle entraîné)
├── data/
│   ├── raw/             # SMSSpamCollection (gitignored)
│   └── processed/       # sms_clean.csv, DATA_CARD.md
├── src/
│   ├── data/prepare.py  # Nettoyage dataset
│   └── model/train.py   # Entraînement TF-IDF + MultinomialNB
├── tests/               # test_api.py, test_contract.py
├── streamlit_app.py     # Frontend
├── requirements.txt     # Dépendances complètes
├── requirements-api.txt # Dépendances API only
├── Dockerfile           # Multi-stage (frontend, api)
├── docker-compose.yml
├── .dockerignore
├── .gitattributes
└── pytest.ini
```

## Équipe

- Aaron
- Coumba
- Marc
- Imad
