"""
Unified Campus FAQ Chatbot Demo
--------------------------------
Combines all three teammates' solutions into one Streamlit interface:
    Method A - TF-IDF + SVM                        (traditional ML)   -> implemented below
    Method B - multilingual-BERT                   (deep learning)    -> implemented below
    Method C - Sentence Transformers + Cosine Sim  (semantic search)  -> implemented below

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data.responses import CONFIDENCE_THRESHOLD, RESPONSES

st.set_page_config(page_title="Campus FAQ Chatbot", page_icon="🎓", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
FEEDBACK_COLUMNS = ["timestamp", "question", "method", "intent", "confidence", "helpful", "comment"]


def save_feedback(question: str, method_name: str, intent: str, confidence: float, helpful: str, comment: str) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not FEEDBACK_PATH.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "method": method_name,
        "intent": intent,
        "confidence": f"{confidence:.2f}",
        "helpful": helpful,
        "comment": comment.replace("\n", " ").strip(),
    }

    with FEEDBACK_PATH.open("a", encoding="utf-8", newline="") as f:
        if is_new_file:
            f.write(",".join(FEEDBACK_COLUMNS) + "\n")
        f.write(",".join(f'"{row[col]}"' for col in FEEDBACK_COLUMNS) + "\n")


def render_feedback_form(form_key: str, question: str, method_name: str, intent: str, confidence: float) -> None:
    with st.form(key=form_key):
        helpful = st.radio("Was this answer helpful?", ["Yes", "No"], horizontal=True, key=f"{form_key}_helpful")
        comment = st.text_input("Optional comment", key=f"{form_key}_comment")
        if st.form_submit_button("Submit feedback"):
            save_feedback(question, method_name, intent, confidence, helpful, comment)
            st.success("Thanks for your feedback!")


# ---------------------------------------------------------------------------
# METHOD A - TF-IDF + SVM  (already working)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_method_a():
    vectorizer_path = MODEL_DIR / "method_a_vectorizer.pkl"
    clf_path = MODEL_DIR / "method_a_svm.pkl"
    label_path = MODEL_DIR / "method_a_label_encoder.pkl"

    missing = [str(path.name) for path in [vectorizer_path, clf_path, label_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing) + ". Run python train_method_a.py first."
        )

    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(clf_path)
    label_encoder = joblib.load(label_path)
    return vectorizer, clf, label_encoder


def predict_method_a(text: str):
    if not isinstance(text, str):
        text = str(text)

    cleaner = " ".join(text.strip().lower().split())
    if not cleaner:
        return "fallback", 0.0

    vectorizer, clf, label_encoder = load_method_a()
    X = vectorizer.transform([cleaner])
    probs = clf.predict_proba(X)[0]
    best_idx = probs.argmax()
    intent = label_encoder.inverse_transform([best_idx])[0]
    confidence = float(probs[best_idx])
    return intent, confidence


# ---------------------------------------------------------------------------
# METHOD B - multilingual-BERT
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None


@st.cache_resource
def load_method_b():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise ImportError("Install transformers and torch to use Method B.")

    model_dir = MODEL_DIR / "method_b_bert"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return tokenizer, model


def predict_method_b(text: str):
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        return "not_implemented", 0.0

    try:
        tokenizer, model = load_method_b()
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        best_idx = int(probs.argmax())
        confidence = float(probs[best_idx])
        intent = model.config.id2label[best_idx]
        return intent, confidence
    except Exception:
        return "not_implemented", 0.0


# ---------------------------------------------------------------------------
# METHOD C - Sentence Transformers + Cosine Similarity (Semantic Search)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:
    _SentenceTransformer = None


@st.cache_resource
def load_method_c():
    if _SentenceTransformer is None:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    emb_path   = MODEL_DIR / "method_c_embeddings.npy"
    label_path = MODEL_DIR / "method_c_labels.pkl"
    missing = [str(p.name) for p in [emb_path, label_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Method C artifacts: " + ", ".join(missing) + ". Run python train_method_c.py first."
        )
    embeddings = np.load(str(emb_path))
    labels     = joblib.load(label_path)
    model      = _SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return embeddings, labels, model


def predict_method_c(text: str):
    if _SentenceTransformer is None:
        return "not_implemented", 0.0
    try:
        from collections import Counter
        embeddings, labels, model = load_method_c()
        vec         = model.encode([text])[0]
        corpus_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        query_norm  = vec / (np.linalg.norm(vec) + 1e-10)
        sims        = corpus_norm @ query_norm
        top_k_idx   = np.argsort(sims)[::-1][:5]
        top_k_labels = [labels[i] for i in top_k_idx]
        intent      = Counter(top_k_labels).most_common(1)[0][0]
        confidence  = float(sims[top_k_idx[0]])
        return intent, confidence
    except Exception:
        return "not_implemented", 0.0




# ---------------------------------------------------------------------------
# Shared helper: turn (intent, confidence) into a chat reply
# ---------------------------------------------------------------------------
def get_reply(intent: str, confidence: float) -> str:
    if confidence < CONFIDENCE_THRESHOLD or intent not in RESPONSES:
        return RESPONSES["fallback"]
    return RESPONSES[intent]


METHODS = {
    "Method A - TF-IDF + SVM (Traditional ML)": predict_method_a,
    "Method B - Multilingual BERT (Deep Learning)": predict_method_b,
    "Method C - Sentence Transformers (Semantic Search)": predict_method_c,
}

EXAMPLE_QUESTIONS = [
    "How do I check my exam timetable?",
    "How to register for courses?",
    "Where can I pay my tuition fees?",
    "How to apply hostel?",
    "How do I access library services?",
    "I can't connect to campus wifi",
]


def render_answer_box(method_name: str, intent: str, confidence: float):
    if intent == "not_implemented":
        if method_name.startswith("Method B"):
            st.warning("Method B model is unavailable. Run `python train_method_b.py` first.")
        elif method_name.startswith("Method C"):
            st.warning("Method C model is unavailable. Run `python train_method_c.py` first.")
        else:
            st.warning("This method is unavailable right now.")
        st.caption("Predicted intent: unavailable | Confidence: 0.00")
        return

    reply = get_reply(intent, confidence)
    st.markdown(f"### {method_name}")
    st.info(reply)
    st.caption(f"Predicted intent: {intent} | Confidence: {confidence:.2f}")



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quick_question" not in st.session_state:
    st.session_state.quick_question = ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎓 Campus FAQ Chatbot")
st.caption("Multilingual (English / Bahasa Malaysia / Mandarin / Manglish) campus assistant")

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Single method", "Compare all 3 methods", "Evaluation"])
selected_method = st.sidebar.selectbox("Choose method", list(METHODS.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Quick examples")
for question in EXAMPLE_QUESTIONS:
    if st.sidebar.button(question, key=f"q_{question}"):
        st.session_state.quick_question = question

if mode == "Single method":
    with st.form(key="single_method_form"):
        prompt = st.text_input(
            "Ask a question about campus services...",
            value=st.session_state.quick_question,
            placeholder="e.g. how do I check my exam timetable?",
        )
        submitted = st.form_submit_button("Send")

    if submitted and prompt.strip():
        st.session_state.quick_question = ""
        predict_fn = METHODS[selected_method]
        intent, confidence = predict_fn(prompt)

        if intent == "not_implemented":
            st.warning("This method is unavailable right now. Check its setup (trained model).")
        else:
            reply = get_reply(intent, confidence)

            st.session_state.chat_history.append({"role": "user", "text": prompt})
            st.session_state.chat_history.append({"role": "assistant", "text": reply})

            st.subheader("Latest response")
            st.markdown(f"**{selected_method.split(' - ')[0]}:** {reply}")
            with st.expander("Prediction details"):
                st.write(f"Intent: `{intent}`")
                st.write(f"Confidence: `{confidence:.2f}`")

            render_feedback_form("single_feedback", prompt, selected_method, intent, confidence)

    if st.session_state.chat_history:
        st.subheader("Conversation history")
        for item in st.session_state.chat_history:
            if item["role"] == "user":
                st.chat_message("user").write(item["text"])
            else:
                st.chat_message("assistant").write(item["text"])

elif mode == "Compare all 3 methods":
    with st.form(key="compare_form"):
        prompt = st.text_input(
            "Ask a question to compare all methods...",
            value=st.session_state.quick_question,
            placeholder="e.g. where can I pay my tuition fee?",
        )
        submitted = st.form_submit_button("Compare")

    if submitted and prompt.strip():
        st.session_state.quick_question = ""
        st.subheader("Side-by-side comparison")
        cols = st.columns(3)
        for idx, (col, (method_name, predict_fn)) in enumerate(zip(cols, METHODS.items())):
            with col:
                intent, confidence = predict_fn(prompt)
                render_answer_box(method_name, intent, confidence)
                if intent not in ("waking_up", "not_implemented"):
                    render_feedback_form(f"compare_feedback_{idx}", prompt, method_name, intent, confidence)
else:
    st.subheader("Evaluation results")
    st.caption("Run the buttons below on demand; nothing here runs unless you click a button.")

    st.markdown("#### Intent classification (Accuracy / Precision / Recall / F1)")
    intent_metrics_path = DATA_DIR / "evaluation_metrics.csv"
    if st.button("Run intent classification evaluation"):
        with st.spinner("Running Method A / B / C on the fixed test set..."):
            from evaluate_methods import run_evaluation

            run_evaluation()
        st.success("Done. Table refreshed below.")
    if intent_metrics_path.exists():
        st.dataframe(pd.read_csv(intent_metrics_path), use_container_width=True)
    else:
        st.info("Not generated yet. Click the button above to run it.")

    st.markdown("#### Response relevancy (BLEU / ROUGE-L)")
    st.caption(
        "Responses are fixed canned text per intent, so these scores mainly reflect whether each method "
        "retrieved the correct intent, not free-text generation quality."
    )
    relevancy_metrics_path = DATA_DIR / "response_relevancy_metrics.csv"
    if st.button("Run response relevancy evaluation"):
        results_path = DATA_DIR / "evaluation_results.csv"
        if not results_path.exists():
            st.warning("Run the intent classification evaluation first (it produces the predictions needed here).")
        else:
            with st.spinner("Scoring BLEU / ROUGE-L..."):
                from evaluate_response_relevancy import run_relevancy_evaluation

                run_relevancy_evaluation()
            st.success("Done. Table refreshed below.")
    if relevancy_metrics_path.exists():
        st.dataframe(pd.read_csv(relevancy_metrics_path), use_container_width=True)
    else:
        st.info("Not generated yet. Click the button above to run it.")

    st.markdown("#### User feedback (helpful %)")
    if FEEDBACK_PATH.exists():
        feedback_df = pd.read_csv(FEEDBACK_PATH)
        if not feedback_df.empty:
            summary = (
                feedback_df.groupby("method")["helpful"]
                .apply(lambda col: (col == "Yes").mean() * 100)
                .reset_index(name="helpful_pct")
            )
            summary["total_responses"] = feedback_df.groupby("method").size().values
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No feedback collected yet. Feedback appears here once users submit the feedback form.")
    else:
        st.info("No feedback collected yet. Feedback appears here once users submit the feedback form.")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Method A: run train_method_a.py | "
    "Method B: run train_method_b.py | "
    "Method C: run train_method_c.py — all fully local, no server needed."
)
