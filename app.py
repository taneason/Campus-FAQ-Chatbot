"""
Unified Campus FAQ Chatbot Demo
--------------------------------
This app combines the working TF-IDF + SVM method with a polished demo UI
for a campus FAQ assistant. It is designed to be assignment-ready and easy
for teammates to extend with new methods.
"""

from pathlib import Path

import joblib
import streamlit as st

from data.responses import CONFIDENCE_THRESHOLD, RESPONSES

st.set_page_config(page_title="Campus FAQ Chatbot", page_icon="🎓", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# METHOD A - working TF-IDF + SVM
# ---------------------------------------------------------------------------
@st.cache_resource
def load_method_a():
    vectorizer_path = MODEL_DIR / "method_a_vectorizer.pkl"
    clf_path = MODEL_DIR / "method_a_svm.pkl"
    label_path = MODEL_DIR / "method_a_label_encoder.pkl"

    missing = [str(path.name) for path in [vectorizer_path, clf_path, label_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing)
            + ". Run python train_method_a.py first."
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
# METHOD B - placeholder for teammate B
# ---------------------------------------------------------------------------
def predict_method_b(text: str):
    return "not_implemented", 0.0


# ---------------------------------------------------------------------------
# METHOD C - placeholder for teammate C
# ---------------------------------------------------------------------------
def predict_method_c(text: str):
    return "not_implemented", 0.0


# ---------------------------------------------------------------------------
# Shared helpers
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

if hasattr(st.session_state, "quick_question"):
    user_input = st.session_state.quick_question
else:
    user_input = ""

if mode == "Single method":
    if user_input:
        prompt = user_input
    else:
        prompt = st.chat_input("Ask a question about campus services...")

    if prompt:
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
    if user_input:
        prompt = user_input
    else:
        prompt = st.chat_input("Ask a question to compare all methods...")

    if prompt:
        st.subheader("Side-by-side comparison")
        cols = st.columns(3)
        for col, (method_name, predict_fn) in zip(cols, METHODS.items()):
            with col:
                intent, confidence = predict_fn(prompt)
                render_answer_box(method_name, intent, confidence)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Method A is fully implemented and trained from the shared FAQ dataset. "
    "Methods B and C are kept as placeholders to be filled in by teammates."
)
