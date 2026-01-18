import streamlit as st
import joblib
import os
import numpy as np

# -----------------------------
# PATH SETUP (RENDER SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

CATEGORY_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "email_classification", "category_lr.pkl"
)
CATEGORY_VECTORIZER_PATH = os.path.join(
    PROJECT_ROOT, "models", "email_classification", "category_tfidf_vectorizer.pkl"
)

URGENCY_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "urgency", "urgency_lr.pkl"
)
URGENCY_VECTORIZER_PATH = os.path.join(
    PROJECT_ROOT, "models", "urgency", "urgency_tfidf_vectorizer.pkl"
)

# -----------------------------
# LOAD MODELS (LIGHTWEIGHT)
# -----------------------------
@st.cache_resource
def load_category_models():
    model = joblib.load(CATEGORY_MODEL_PATH)
    vectorizer = joblib.load(CATEGORY_VECTORIZER_PATH)
    return model, vectorizer

@st.cache_resource
def load_urgency_models():
    model = joblib.load(URGENCY_MODEL_PATH)
    vectorizer = joblib.load(URGENCY_VECTORIZER_PATH)
    return model, vectorizer

# -----------------------------
# RULE-BASED FALLBACK
# -----------------------------
def rule_based_urgency(text):
    text = text.lower()
    if any(k in text for k in ["urgent", "asap", "immediately", "down", "failure"]):
        return "high"
    elif any(k in text for k in ["soon", "delay", "issue"]):
        return "medium"
    return "low"

# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------
def predict_category(text):
    model, vectorizer = load_category_models()
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    label = model.classes_[np.argmax(probs)]
    return label, float(np.max(probs))

def predict_urgency(text):
    model, vectorizer = load_urgency_models()
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    label = model.classes_[np.argmax(probs)]
    conf = float(np.max(probs))

    if conf < 0.6:
        label = rule_based_urgency(text)

    return label, conf

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Email Classifier", layout="centered")

st.title("📧 AI-Based Email Classifier")
st.caption("Lightweight ML system with rule-based fallback")

email_text = st.text_area("Enter email content")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email text.")
    else:
        with st.spinner("Classifying email..."):
            category, cat_conf = predict_category(email_text)
            urgency, urg_conf = predict_urgency(email_text)

        st.success("Prediction Complete")
        st.markdown(f"**Category:** `{category}` (confidence: {cat_conf:.2f})")
        st.markdown(f"**Urgency:** `{urgency}` (confidence: {urg_conf:.2f})")
