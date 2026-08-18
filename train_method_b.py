"""
Train Method B (multilingual BERT intent classifier).

Usage:
    python train_method_b.py
    python train_method_b.py --epochs 4 --batch-size 16 --lr 2e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
MODEL_NAME = "bert-base-multilingual-cased"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "faq_data.csv"
OUTPUT_DIR = BASE_DIR / "models" / "method_b_bert"
REPORT_PATH = BASE_DIR / "models" / "method_b_eval_report.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multilingual BERT for Method B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    return labels, preds


def main():
    args = parse_args()
    set_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "intent"])

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = IntentDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=args.max_length,
    )
    val_ds = IntentDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training on device: {device}")
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        y_true, y_pred = evaluate(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch}/{args.epochs} - train_loss={avg_loss:.4f} - val_acc={val_acc:.4f}")

    y_true, y_pred = evaluate(model, val_loader, device)
    final_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    report_text = (
        "Method B: multilingual-BERT Evaluation\n"
        f"Base model: {MODEL_NAME}\n"
        f"Validation accuracy: {final_acc:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("Training complete.")
    print(f"Saved model + tokenizer: {OUTPUT_DIR}")
    print(f"Saved eval report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
