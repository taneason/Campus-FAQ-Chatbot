"""
Method C - Semantic Similarity: Sentence Transformers + Cosine Similarity
--------------------------------------------------------------------------
No server needed. Encodes all FAQ training examples into sentence embeddings,
then at inference time finds the nearest neighbour by cosine similarity.

Run once to build the index:
    python train_method_c.py

Produces (saved into models/):
    - method_c_embeddings.npy    (matrix of shape [N, dim])
    - method_c_labels.pkl        (list of intent strings, length N)
    - method_c_eval_report.txt   (accuracy / precision / recall / F1)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH  = Path("data/faq_data.csv")
MODEL_DIR  = Path("models")

# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarities between one query row and every corpus row."""
    query_norm  = query_vec  / (np.linalg.norm(query_vec)  + 1e-10)
    corpus_norm = corpus_matrix / (np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10)
    return corpus_norm @ query_norm   # shape: (N,)


def predict(text: str, embeddings: np.ndarray, labels: list, model: SentenceTransformer):
    vec  = model.encode([text])[0]
    sims = cosine_similarity_matrix(vec, embeddings)
    best = int(np.argmax(sims))
    return labels[best], float(sims[best])


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
