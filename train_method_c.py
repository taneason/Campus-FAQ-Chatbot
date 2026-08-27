"""
Method C - Semantic Similarity: Sentence Transformers + Cosine Similarity
--------------------------------------------------------------------------
Improvements over v1:
  - Upgraded model: paraphrase-multilingual-mpnet-base-v2 (stronger than MiniLM)
  - Top-K majority voting (k=5): picks the most common intent among the 5
    nearest neighbours, reducing single-outlier errors

Run once to build the index:
    python train_method_c.py

Produces (saved into models/):
    - method_c_embeddings.npy
    - method_c_labels.pkl
    - method_c_eval_report.txt
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TOP_K      = 5   # majority-vote over top-K neighbours
DATA_PATH  = Path("data/faq_data.csv")
MODEL_DIR  = Path("models")

def cosine_similarity_matrix(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    query_norm  = query_vec  / (np.linalg.norm(query_vec)  + 1e-10)
    corpus_norm = corpus_matrix / (np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10)
    return corpus_norm @ query_norm


def predict(text: str, embeddings: np.ndarray, labels: list, model: SentenceTransformer, k: int = TOP_K):
    vec  = model.encode([text])[0]
    sims = cosine_similarity_matrix(vec, embeddings)
    top_k_idx = np.argsort(sims)[::-1][:k]
    # Majority vote among top-k neighbours; tie-break by highest similarity
    top_k_labels = [labels[i] for i in top_k_idx]
    intent = Counter(top_k_labels).most_common(1)[0][0]
    confidence = float(sims[top_k_idx[0]])
    return intent, confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "intent"])
    print(f"Loaded {len(df)} examples across {df['intent'].nunique()} intents")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["intent"]
    )

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # Build corpus index from training split
    print("Encoding training corpus ...")
    train_texts  = train_df["text"].tolist()
    train_labels = train_df["intent"].tolist()
    train_embs   = model.encode(train_texts, show_progress_bar=True, batch_size=32)

    # Evaluate on test split
    print("Evaluating on test split ...")
    test_texts  = test_df["text"].tolist()
    test_labels = test_df["intent"].tolist()

    preds = []
    for text in test_texts:
        intent, _ = predict(text, train_embs, train_labels, model)
        preds.append(intent)

    acc    = accuracy_score(test_labels, preds)
    report = classification_report(
        test_labels, preds,
        labels=sorted(df["intent"].unique()),
        zero_division=0
    )
    cm = confusion_matrix(test_labels, preds, labels=sorted(df["intent"].unique()))

    print(f"\nAccuracy: {acc:.3f}\n")
    print(report)

    # Save model artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    np.save(MODEL_DIR / "method_c_embeddings.npy", train_embs)
    joblib.dump(train_labels, MODEL_DIR / "method_c_labels.pkl")

    report_text = (
        f"Method C - Sentence Transformers ({MODEL_NAME})\n"
        f"Accuracy: {acc:.3f}\n\n"
        f"{report}\n"
        f"Confusion Matrix (rows=true, cols=pred):\n{cm}\n"
    )
    (MODEL_DIR / "method_c_eval_report.txt").write_text(report_text, encoding="utf-8")

    print(f"Saved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
