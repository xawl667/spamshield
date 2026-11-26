# Data Card — SMS Spam Collection (SpamShield)

## Source et droits
- **Dataset** : SMS Spam Collection v.1
- **Dépôt** : UCI Machine Learning Repository
- **URL** : https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- **Auteurs** : T. Almeida et J. M. Gómez Hidalgo (2011)
- **Citation** : Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
- **Licence** : CC BY 4.0 (réutilisation libre avec obligation de citer la source)

## Fichiers
- **Brut** : `data/raw/SMSSpamCollection` — format `label<tabulation>texte`, labels `ham` / `spam`
- **Préparé** : `data/processed/sms_clean.csv` — colonnes `text` et `label`, label ∈ {`spam`, `legitime`}

## Nettoyage (chiffres réels)

### Données brutes
- **Lignes brutes** : 5 574
- **Répartition brute** :
  - `ham` : 4 827
  - `spam` : 747

### Opérations de nettoyage (d'après `src/data/prepare.py`)

| Étape | Code (lignes) | Résultat |
|-------|---------------|----------|
| Lecture CSV | `pd.read_csv(sep="\t", header=None, names=["label", "text"], encoding="utf-8", quoting=3)` (l. 27-33) | 5 574 lignes chargées |
| Suppression NaN | `df.dropna(subset=["label", "text"])` (l. 37) | 0 lignes supprimées (pas de NaN) |
| Strip texte | `df["text"] = df["text"].astype(str).str.strip()` (l. 38) | — |
| Strip + minuscules label | `df["label"] = df["label"].astype(str).str.strip().str.lower()` (l. 39) | — |
| Suppression textes vides | `df = df[df["text"] != ""]` (l. 41) | **0** lignes supprimées (aucun texte vide après strip) |
| Renommage labels | `label_mapping = {"ham": "legitime", "spam": "spam"}` + `map()` (l. 43-44) | `ham` → `legitime`, `spam` inchangé |
| Suppression NaN post-mapping | `df.dropna(subset=["label"])` (l. 45) | 0 lignes supprimées |
| Suppression doublons exacts | `df.drop_duplicates(subset=["text"], keep="first")` (l. 47) | **414** lignes supprimées (289 textes dupliqués, keep=first) |
| Sélection colonnes | `df[["text", "label"]]` (l. 48) | Colonnes finales : `text`, `label` |

### Détail des doublons
- **Textes dupliqués (groupes uniques)** : 289
- **Lignes totales concernées** : 703 (toutes occurrences)
- **Lignes supprimées** : 414 (= 703 - 289, une occurrence gardée par groupe)
- **Conflits de labels** (même texte, labels différents) : **0** (vérifié)

### Données finales
- **Lignes finales** : 5 160
- **Répartition finale** :
  - `legitime` : 4 518
  - `spam` : 642
- **Pourcentage de spam** : 12,4 %

## Limites connues
- **Langue** : anglais uniquement (forum britannique + SMS d'étudiants de Singapour)
- **Période** : données de 2011 (technologie SMS, pas de messageries instantanées modernes, pas de liens raccourcis type bit.ly)
- **Déséquilibre de classes** : 12,4 % de spam (environ 1:7)
- **Biais géographiques/culturels** : corpus collecté au Royaume-Uni et à Singapour, ne reflète pas la diversité mondiale actuelle
- **Absence de spam récent** : pas de smishing (phishing par SMS), pas de spam via applications de messagerie (WhatsApp, Telegram, etc.)
- **Prétraitement minimal** : pas de normalisation Unicode avancée, pas de détection de langues, conservation de la casse dans le texte (le vectoriseur TF-IDF gère `lowercase=True`)

## Génération
- Script : `src/data/prepare.py`
- Commande : `python src/data/prepare.py`
- Date de génération : 2026-11-26
- Artefacts produits : `data/processed/sms_clean.csv`, `data/processed/class_distribution.png`