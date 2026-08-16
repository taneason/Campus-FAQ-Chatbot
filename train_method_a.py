"""
Method A - Traditional ML: TF-IDF + SVM intent classifier
-----------------------------------------------------------
Run this once to train and save the model:
    python train_method_a.py

Produces (saved into models/):
    - method_a_vectorizer.pkl
    - method_a_svm.pkl
    - method_a_label_encoder.pkl
    - method_a_eval_report.txt   (accuracy / precision / recall / F1 for your documentation)
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = "data/faq_data.csv"
MODEL_DIR = "models"


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} examples across {df['intent'].nunique()} intents")

    # 2. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["intent"])

    # 3. Split (stratify keeps intent balance in both sets even with a small dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. TF-IDF vectorization
    #    ngram_range=(1,2) helps catch short Manglish phrases like "check exam timetable"
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train SVM (probability=True so we can show a confidence score in the demo)
    clf = SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(X_train_vec, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.3f}\n")
    print(report)

    with open(f"{MODEL_DIR}/method_a_eval_report.txt", "w") as f:
        f.write(f"Method A - TF-IDF + SVM\nAccuracy: {acc:.3f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(str(cm))

    # 7. Save artifacts
    joblib.dump(vectorizer, f"{MODEL_DIR}/method_a_vectorizer.pkl")
    joblib.dump(clf, f"{MODEL_DIR}/method_a_svm.pkl")
    joblib.dump(label_encoder, f"{MODEL_DIR}/method_a_label_encoder.pkl")

    print(f"\nSaved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
