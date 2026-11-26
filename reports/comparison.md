# Comparaison Baseline vs Modèle TF-IDF + MultinomialNB

**Source** : `reports/metrics.json` (répertoire de travail)
**Versions mentionnées** : Python 3.13.3, scikit-learn 1.9.1
**Split** : 80/20 stratifié, `random_state=42` — Train: 4 128, Test: 1 032 (Legitime: 904, Spam: 128)

---

## Métriques classe `spam` (classe positive)

| Métrique | Baseline (DummyClassifier<br>`most_frequent`) | TF-IDF + MultinomialNB |
|----------|-----------------------------------------------|------------------------|
| **Précision (Precision)** | 0,0000 | **0,9746** |
| **Rappel (Recall)** | 0,0000 | **0,8984** |
| **F1-score** | 0,0000 | **0,9350** |
| **Accuracy** | 0,8760 | **0,9845** |

---

## Matrice de confusion (jeu de test : 904 legitime, 128 spam)

### Baseline (`most_frequent` → prédit toujours `legitime`)
| | Prédit `legitime` | Prédit `spam` |
|---|---|---|
| **Réel `legitime`** | 904 (VN) | 0 (FP) |
| **Réel `spam`** | 128 (FN) | 0 (VP) |

- **Faux positifs** (légitime classé spam) : 0
- **Faux négatifs** (spam non détecté) : 128

### TF-IDF + MultinomialNB
| | Prédit `legitime` | Prédit `spam` |
|---|---|---|
| **Réel `legitime`** | 901 (VN) | 3 (FP) |
| **Réel `spam`** | 13 (FN) | 115 (VP) |

- **Faux positifs** (légitime classé spam) : **3**
- **Faux négatifs** (spam non détecté) : **13**

---

## Paramètres du modèle
- **TF-IDF** : `lowercase=True`, `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`
- **MultinomialNB** : `alpha=0.1`
- **Baseline** : `DummyClassifier(strategy="most_frequent")`
- **Vocabulaire TF-IDF** : 10 050 termes
- **Classes du pipeline** : `["legitime", "spam"]`

---

## Note
> Hyperparamètres fixés a priori, non réglés sur le test.