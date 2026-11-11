from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_pipeline.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Keep train_model.py in the same folder as spam.csv."
        )

    # The supplied dataset is commonly saved in Latin-1 encoding.
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # The supplied dataset uses v1 for the label and v2 for the message.
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
    elif {"label", "message"}.issubset(df.columns):
        df = df[["label", "message"]].copy()
    else:
        raise ValueError(f"Unexpected dataset columns: {list(df.columns)}")

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["label", "message"])

    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # A single pipeline keeps preprocessing/vectorization identical at training and prediction time.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Not Spam", "Spam"],
            output_dict=True, zero_division=0
        )
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nMODEL TRAINED SUCCESSFULLY")
    print(f"Clean dataset rows : {len(df)}")
    print(f"Training messages : {len(X_train)}")
    print(f"Testing messages  : {len(X_test)}")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"F1-score          : {metrics['f1_score']:.4f}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions))
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved metrics: {METRICS_PATH.name}")

if __name__ == "__main__":
    main()
