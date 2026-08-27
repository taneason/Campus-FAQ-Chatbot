"""
Evaluate Method A, B, and C on a fixed held-out test set.

Usage:
    python evaluate_methods.py

Writes:
    data/evaluation_results.csv   (per-question predictions for all methods)
    models/evaluation_report.txt  (accuracy / precision / recall / F1 comparison table)
"""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEST_SET_PATH = BASE_DIR / "data" / "test_set.csv"
RESULTS_PATH = BASE_DIR / "data" / "evaluation_results.csv"
REPORT_PATH = BASE_DIR / "models" / "evaluation_report.txt"
METRICS_CSV_PATH = BASE_DIR / "data" / "evaluation_metrics.csv"


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------
def load_method_a():
    vectorizer = joblib.load(MODEL_DIR / "method_a_vectorizer.pkl")
    clf = joblib.load(MODEL_DIR / "method_a_svm.pkl")
    label_encoder = joblib.load(MODEL_DIR / "method_a_label_encoder.pkl")
    return vectorizer, clf, label_encoder


def predict_method_a(text: str, artifacts) -> str:
    vectorizer, clf, label_encoder = artifacts
    cleaner = " ".join(text.strip().lower().split())
    X = vectorizer.transform([cleaner])
    probs = clf.predict_proba(X)[0]
    best_idx = probs.argmax()
    return label_encoder.inverse_transform([best_idx])[0]


# ---------------------------------------------------------------------------
# Method B
# ---------------------------------------------------------------------------
def load_method_b():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = MODEL_DIR / "method_b_bert"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return tokenizer, model


def predict_method_b(text: str, artifacts) -> str:
    import torch

    tokenizer, model = artifacts
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    best_idx = int(probs.argmax())
    return model.config.id2label[best_idx]


# ---------------------------------------------------------------------------
# Method C - Sentence Transformers + Cosine Similarity
# ---------------------------------------------------------------------------
def load_method_c():
    from collections import Counter
    from sentence_transformers import SentenceTransformer

    embeddings = np.load(str(MODEL_DIR / "method_c_embeddings.npy"))
    labels = joblib.load(MODEL_DIR / "method_c_labels.pkl")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return embeddings, labels, model, Counter


def predict_method_c(text: str, artifacts) -> str:
    embeddings, labels, model, Counter = artifacts
    vec = model.encode([text])[0]
    corpus_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    query_norm = vec / (np.linalg.norm(vec) + 1e-10)
    sims = corpus_norm @ query_norm
    top_k_idx = np.argsort(sims)[::-1][:5]
    top_k_labels = [labels[i] for i in top_k_idx]
    return Counter(top_k_labels).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, label: str) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"{label}: accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    return {"method": label, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_evaluation() -> list[dict]:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Test set not found: {TEST_SET_PATH}")

    df = pd.read_csv(TEST_SET_PATH)
    questions = df["text"].tolist()
    y_true = df["intent"].tolist()

    results_rows = []
    metrics = []

    # Method A
    a_artifacts = load_method_a()
    preds_a = [predict_method_a(q, a_artifacts) for q in questions]
    metrics.append(evaluate(y_true, preds_a, "Method A (TF-IDF + SVM)"))

    # Method B
    preds_b = []
    try:
        b_artifacts = load_method_b()
        preds_b = [predict_method_b(q, b_artifacts) for q in questions]
        metrics.append(evaluate(y_true, preds_b, "Method B (multilingual BERT)"))
    except Exception as exc:
        print(f"Skipping Method B: {exc}")
        preds_b = ["not_implemented"] * len(questions)

    # Method C
    preds_c = []
    try:
        c_artifacts = load_method_c()
        preds_c = [predict_method_c(q, c_artifacts) for q in questions]
        metrics.append(evaluate(y_true, preds_c, "Method C (Sentence Transformers)"))
    except Exception as exc:
        print(f"Skipping Method C: {exc}")
        preds_c = ["not_implemented"] * len(questions)

    for i, question in enumerate(questions):
        results_rows.append(
            {
                "question": question,
                "expected_intent": y_true[i],
                "method_a_pred": preds_a[i],
                "method_b_pred": preds_b[i],
                "method_c_pred": preds_c[i],
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    report_lines = ["Method Comparison - Intent Classification Evaluation", ""]
    report_lines.append(f"Test set size: {len(questions)} questions\n")
    report_lines.append(f"{'Method':<35}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    for m in metrics:
        report_lines.append(
            f"{m['method']:<35}{m['accuracy']:>10.4f}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}"
        )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    with METRICS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\nSaved per-question predictions: {RESULTS_PATH}")
    print(f"Saved comparison report: {REPORT_PATH}")
    print(f"Saved metrics CSV: {METRICS_CSV_PATH}")
    return metrics


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
