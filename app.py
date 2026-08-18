"""
Unified Campus FAQ Chatbot Demo
--------------------------------
Combines all three teammates' solutions into one Streamlit interface:
    Method A - TF-IDF + SVM              (traditional ML)      -> implemented below
    Method B - multilingual-BERT         (deep learning)       -> implemented below
    Method C - Rasa                      (platform-based)      -> implemented below

Run with:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import requests
import streamlit as st

from data.responses import CONFIDENCE_THRESHOLD, RESPONSES

st.set_page_config(page_title="Campus FAQ Chatbot", page_icon="🎓", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"


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
# METHOD C - Rasa
# ---------------------------------------------------------------------------
RASA_URL = "http://localhost:5005/model/parse"


def predict_method_c(text: str):
    try:
        resp = requests.post(RASA_URL, json={"text": text}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        intent = data["intent"]["name"]
        confidence = float(data["intent"]["confidence"])
        return intent, confidence
    except (requests.RequestException, KeyError, TypeError, ValueError):
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
    "Method C - Rasa (Platform-based)": predict_method_c,
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
mode = st.sidebar.radio("Mode", ["Single method", "Compare all 3 methods"])
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
        reply = get_reply(intent, confidence)

        st.session_state.chat_history.append({"role": "user", "text": prompt})
        st.session_state.chat_history.append({"role": "assistant", "text": reply})

        st.subheader("Latest response")
        st.markdown(f"**{selected_method.split(' - ')[0]}:** {reply}")
        with st.expander("Prediction details"):
            st.write(f"Intent: `{intent}`")
            st.write(f"Confidence: `{confidence:.2f}`")

    if st.session_state.chat_history:
        st.subheader("Conversation history")
        for item in st.session_state.chat_history:
            if item["role"] == "user":
                st.chat_message("user").write(item["text"])
            else:
                st.chat_message("assistant").write(item["text"])

else:
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
        for col, (method_name, predict_fn) in zip(cols, METHODS.items()):
            with col:
                intent, confidence = predict_fn(prompt)
                render_answer_box(method_name, intent, confidence)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Method A is fully implemented and trained. Method B requires a local BERT model in models/method_b_bert; "
    "Method C expects a local Rasa server at http://localhost:5005."
)
