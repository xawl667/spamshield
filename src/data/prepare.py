import io
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import requests

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_FILE = RAW_DIR / "SMSSpamCollection"
CLEAN_CSV = PROCESSED_DIR / "sms_clean.csv"
CHART_PATH = PROCESSED_DIR / "class_distribution.png"


def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        return

    response = requests.get(UCI_ZIP_URL, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extract("SMSSpamCollection", path=RAW_DIR)


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        quoting=3,
    )

    initial_count = len(df)

    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]

    label_mapping = {"ham": "legitime", "spam": "spam"}
    df["label"] = df["label"].map(label_mapping)
    df = df.dropna(subset=["label"])

    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    df = df[["text", "label"]]

    final_count = len(df)
    print(f"Lignes brutes : {initial_count}")
    print(f"Lignes supprimées (vides/doublons) : {initial_count - final_count}")
    print(f"Lignes finales : {final_count}")

    return df


def plot_distribution(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    counts = df["label"].value_counts()
    percentages = df["label"].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5)

    ax.set_title("Distribution des classes - SMS Spam Collection (UCI)", fontsize=12, pad=12)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre de SMS")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (counts.max() * 0.02),
            f"{count} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()


def validate_contract(df: pd.DataFrame) -> None:
    assert list(df.columns) == ["text", "label"], f"Colonnes invalides : {list(df.columns)}"
    assert df["text"].isna().sum() == 0, "Valeurs manquantes détectées dans 'text'"
    assert (df["text"].str.strip() == "").sum() == 0, "Textes vides détectés"
    assert set(df["label"].unique()) == {"legitime", "spam"}, f"Labels invalides : {df['label'].unique()}"
    assert CLEAN_CSV.exists(), "Le fichier sms_clean.csv n'a pas été généré"
    print("Contrat ML validé avec succès.")


def main() -> None:
    download_dataset()
    df_clean = load_and_clean()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    plot_distribution(df_clean)
    validate_contract(df_clean)


if __name__ == "__main__":
    main()
