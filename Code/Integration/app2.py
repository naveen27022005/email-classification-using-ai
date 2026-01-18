import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import os

# --------------------------------------------------
# GLOBAL SETTINGS (LOW MEMORY)
# --------------------------------------------------
torch.set_grad_enabled(False)

st.set_page_config(
    page_title="AI Email Classifier",
    page_icon="📧",
    layout="centered"
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
CATEGORY_MODEL_HF = "naveen-27022005/distilbert-email-category-classifier"

CATEGORY_LABELS = [
    "complaint",
    "feedback",
    "ham",
    "other",
    "request",
    "spam"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

URGENCY_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "urgency", "urgency_lr.pkl"
)

URGENCY_VECTORIZER_PATH = os.path.join(
    PROJECT_ROOT, "models", "urgency", "urgency_tfidf_vectorizer.pkl"
)

# --------------------------------------------------
# LOAD URGENCY MODELS (LIGHTWEIGHT)
# --------------------------------------------------
@st.cache_resource
def load_urgency_models():
    lr_model = joblib.load(URGENCY_MODEL_PATH)
    vectorizer = joblib.load(URGENCY_VECTORIZER_PATH)
    return lr_model, vectorizer


# --------------------------------------------------
# LOAD CATEGORY MODEL (HEAVY – LAZY LOADED)
# --------------------------------------------------
@st.cache_resource
def load_category_model():
    tokenizer = AutoTokenizer.from_pretrained(CATEGORY_MODEL_HF)
    model = AutoModelForSequenceClassification.from_pretrained(CATEGORY_MODEL_HF)
    model.eval()
    return tokenizer, model


# --------------------------------------------------
# RULE-BASED FALLBACK (CATEGORY)
# --------------------------------------------------
def rule_based_category(text):
    text = text.lower()

    if any(word in text for word in ["refund", "not working", "issue", "problem", "error", "failed"]):
        return "complaint"
    if any(word in text for word in ["request", "please", "could you", "can you"]):
        return "request"
    if any(word in text for word in ["thank", "appreciate", "feedback"]):
        return "feedback"
    if any(word in text for word in ["buy now", "free", "click", "offer"]):
        return "spam"

    return "other"


# --------------------------------------------------
# URGENCY PREDICTION
# --------------------------------------------------
def predict_urgency(text):
    lr_model, vectorizer = load_urgency_models()
    X = vectorizer.transform([text])
    probs = lr_model.predict_proba(X)[0]
    idx = np.argmax(probs)
    labels = lr_model.classes_
    return labels[idx], probs[idx]


# --------------------------------------------------
# CATEGORY PREDICTION (DISTILBERT)
# --------------------------------------------------
def predict_category(text):
    try:
        tokenizer, model = load_category_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        idx = torch.argmax(probs, dim=1).item()
        return CATEGORY_LABELS[idx], probs[0][idx].item()

    except Exception:
        # Fallback if memory spikes
        return rule_based_category(text), 0.50


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("📧 AI Email Classifier")
st.caption("Hybrid ML system with rule-based fallback")

email_text = st.text_area(
    "Enter email content",
    height=200,
    placeholder="Paste the email text here..."
)

if st.button("🔍 Predict"):
    if not email_text.strip():
        st.warning("Please enter email content.")
    else:
        # CATEGORY
        with st.spinner("Classifying email category..."):
            category, cat_conf = predict_category(email_text)

        # URGENCY
        urgency, urg_conf = predict_urgency(email_text)

        st.success(f"📂 **Category:** {category}  \nConfidence: {cat_conf:.2f}")
        st.info(f"⚡ **Urgency:** {urgency}  \nConfidence: {urg_conf:.2f}")

        st.markdown("---")
        st.caption("DistilBERT loaded lazily • Logistic Regression for urgency")
