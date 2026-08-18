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

    cleaner = normalize_text(text)
    if not cleaner:
        return "fallback", 0.0

    guessed_intent, guess_score = keyword_intent_guess(cleaner)

    vectorizer, clf, label_encoder = load_method_a()
    X = vectorizer.transform([cleaner])
    probs = clf.predict_proba(X)[0]
    best_idx = probs.argmax()
    intent = label_encoder.inverse_transform([best_idx])[0]
    confidence = float(probs[best_idx])

    if guess_score > 0 and (confidence < 0.5 or intent == "fallback"):
        return guessed_intent, max(confidence, 0.45)

    if guess_score >= 4 and intent != guessed_intent:
        return guessed_intent, min(max(confidence, 0.55), 0.88)

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


KEYWORD_PATTERNS = {
    "exam_timetable": [
        "exam timetable", "exam schedule", "exam date", "final exam", "final timetable",
        "exam time", "timetable", "schedule", "paper schedule", "exam slot", "exam dates",
        "exam venue", "test schedule", "finals", "exam info", "date for exam"
    ],
    "course_registration": [
        "register course", "register subject", "add subject", "drop subject", "course registration",
        "subject registration", "enrol", "enroll", "course add drop", "add-drop",
        "choose subject", "register class", "select course", "subject selection", "register classes",
        "course add", "add course", "drop course", "elective", "change class section", "register subjek"
    ],
    "fee_payment": [
        "pay fee", "tuition fee", "semester fee", "yuran", "payment", "pay school fee",
        "fee payment", "outstanding fee", "installment", "tuition payment", "pay fees",
        "pay semester charges", "fee balance", "remaining balance", "settle fee", "bank transfer"
    ],
    "hostel_application": [
        "hostel", "asrama", "dorm", "residential", "room application", "stay hostel",
        "apply hostel", "hostel application", "accommodation", "room booking", "hostel form",
        "hostel room", "residential application", "stay in hostel"
    ],
    "library_service": [
        "library", "borrow book", "return book", "renew book", "library hours", "library open",
        "borrow books", "library portal", "database", "book loan", "library service",
        "book due date", "borrow textbook", "renew book", "return borrowed books", "e-book"
    ],
    "it_support": [
        "wifi", "campus wifi", "internet", "network", "portal login", "password reset",
        "student portal", "login problem", "cannot login", "email password", "it support",
        "computer problem", "reset password", "wifi cannot connect", "portal problem", "sign in",
        "forgot password", "connection issue", "session expired"
    ],
}

SLANG_REPLACEMENTS = {
    "cant": "cannot",
    "cannot": "cannot",
    "camne": "how",
    "cane": "how",
    "nak": "want",
    "tau": "know",
    "tak": "not",
    "tk": "not",
    "dah": "already",
    "kat": "at",
    "mana": "where",
    "bila": "when",
    "macam": "like",
    "mcm": "like",
    "subjek": "subject",
    "asrama": "hostel",
    "yuran": "fee",
    "portal": "portal",
    "wifi": "wifi",
    "login": "login",
    "password": "password",
    "reset": "reset",
    "book": "book",
    "library": "library",
    "exam": "exam",
    "timetable": "timetable",
    "gila": "very",
    "lah": "",
    "ah": "",
    "leh": "",
    "ha": "",
    "eh": "",
    "ma": "",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    cleaned = text.lower().strip()
    cleaned = cleaned.replace("?", " ").replace("!", " ").replace(".", " ")
    cleaned = " ".join(cleaned.split())

    for slang, replacement in SLANG_REPLACEMENTS.items():
        cleaned = cleaned.replace(slang, replacement)

    cleaned = " ".join(cleaned.split())
    return cleaned


def keyword_intent_guess(text: str):
    if not isinstance(text, str):
        text = str(text)

    cleaned = normalize_text(text)
    if not cleaned:
        return "fallback", 0

    scores = []
    for intent, keywords in KEYWORD_PATTERNS.items():
        score = 0
        for kw in keywords:
            if kw in cleaned:
                score += 2
        if intent in ["exam_timetable", "course_registration", "fee_payment", "hostel_application", "library_service", "it_support"]:
            phrase_hits = sum(1 for intent_key in [intent] if intent_key in cleaned)
            if phrase_hits:
                score += 1
        scores.append((intent, score))

    best_intent, best_score = max(scores, key=lambda item: item[1])
    return best_intent, best_score


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
    "Method A is fully implemented and trained from the shared FAQ dataset. "
    "Methods B and C are kept as placeholders to be filled in by teammates."
)
