"""
Unified Campus FAQ Chatbot Demo
--------------------------------
Combines all three teammates' solutions into one Streamlit interface:
    Method A - TF-IDF + SVM              (traditional ML)      -> implemented below
    Method B - multilingual-BERT         (deep learning)       -> TODO for teammate B
    Method C - Rasa                      (platform-based)      -> TODO for teammate C

Run with:
    streamlit run app.py
"""

import streamlit as st
import joblib
import requests
from data.responses import RESPONSES, CONFIDENCE_THRESHOLD

st.set_page_config(page_title="Campus FAQ Chatbot", page_icon="🎓", layout="centered")

MODEL_DIR = "models"


# ---------------------------------------------------------------------------
# METHOD A - TF-IDF + SVM  (already working)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_method_a():
    vectorizer = joblib.load(f"{MODEL_DIR}/method_a_vectorizer.pkl")
    clf = joblib.load(f"{MODEL_DIR}/method_a_svm.pkl")
    label_encoder = joblib.load(f"{MODEL_DIR}/method_a_label_encoder.pkl")
    return vectorizer, clf, label_encoder


def predict_method_a(text: str):
    vectorizer, clf, label_encoder = load_method_a()
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    best_idx = probs.argmax()
    intent = label_encoder.inverse_transform([best_idx])[0]
    confidence = float(probs[best_idx])
    return intent, confidence


# ---------------------------------------------------------------------------
# METHOD B - multilingual-BERT  (TODO: teammate B fills this in)
# ---------------------------------------------------------------------------
# Suggested approach:
#   1. Fine-tune a pretrained model (e.g. "bert-base-multilingual-cased" or
#      "xlm-roberta-base") on the same data/faq_data.csv, save it into
#      models/method_b_bert/ with model.save_pretrained() / tokenizer.save_pretrained()
#   2. Load it here with @st.cache_resource, same pattern as load_method_a()
#   3. Return (intent, confidence) in the same format as predict_method_a()
#
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
#
# @st.cache_resource
# def load_method_b():
#     tokenizer = AutoTokenizer.from_pretrained("models/method_b_bert")
#     model = AutoModelForSequenceClassification.from_pretrained("models/method_b_bert")
#     return tokenizer, model
#
# def predict_method_b(text: str):
#     tokenizer, model = load_method_b()
#     inputs = tokenizer(text, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     probs = torch.softmax(logits, dim=1)[0]
#     best_idx = int(probs.argmax())
#     confidence = float(probs[best_idx])
#     intent = model.config.id2label[best_idx]
#     return intent, confidence

def predict_method_b(text: str):
    return "not_implemented", 0.0


# ---------------------------------------------------------------------------
# METHOD C - Rasa  (TODO: teammate C fills this in)
# ---------------------------------------------------------------------------
# Suggested approach:
#   1. Build the Rasa project separately (nlu.yml, domain.yml, stories.yml)
#      using the same intents as data/faq_data.csv so all three methods
#      are comparable.
#   2. Run the Rasa server locally:
#        rasa run --enable-api --cors "*" --port 5005
#   3. Call its REST API from here (Rasa exposes /model/parse for NLU-only,
#      or /webhooks/rest/webhook for full conversation).
#
# RASA_URL = "http://localhost:5005/model/parse"
#
# def predict_method_c(text: str):
#     resp = requests.post(RASA_URL, json={"text": text}, timeout=5)
#     data = resp.json()
#     intent = data["intent"]["name"]
#     confidence = data["intent"]["confidence"]
#     return intent, confidence

def predict_method_c(text: str):
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

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎓 Campus FAQ Chatbot")
st.caption("Multilingual (English / Bahasa Malaysia / Mandarin / Manglish) intent-based FAQ assistant")

mode = st.sidebar.radio("Mode", ["Single method", "Compare all 3 methods"])

if mode == "Single method":
    method_name = st.sidebar.selectbox("Choose method", list(METHODS.keys()))
    user_input = st.text_input("Ask a question:", placeholder="e.g. macam mana nak check exam timetable")

    if user_input:
        predict_fn = METHODS[method_name]
        intent, confidence = predict_fn(user_input)
        reply = get_reply(intent, confidence)

        st.markdown(f"**Chatbot ({method_name.split(' - ')[0]}):** {reply}")
        with st.expander("Details"):
            st.write(f"Predicted intent: `{intent}`")
            st.write(f"Confidence: `{confidence:.2f}`")

else:
    user_input = st.text_input("Ask a question:", placeholder="e.g. exam schedule where got ah")

    if user_input:
        st.subheader("Side-by-side comparison")
        cols = st.columns(3)
        for col, (method_name, predict_fn) in zip(cols, METHODS.items()):
            with col:
                st.markdown(f"**{method_name.split(' - ')[0]}**")
                st.caption(method_name.split(" - ")[1])
                intent, confidence = predict_fn(user_input)
                reply = get_reply(intent, confidence)
                st.info(reply)
                st.caption(f"intent: `{intent}` | confidence: `{confidence:.2f}`")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Method A is fully implemented. Methods B and C have placeholder "
    "functions ready to be filled in by teammates (see app.py)."
)
