import os
import joblib
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Smart Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 AI-Powered Smart Email Classifier")
st.write("Hybrid ML system with rule-based fallback")

# -------------------------------
# Path Handling (ROBUST)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

CATEGORY_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "distilbert_classifier")
URGENCY_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "urgency", "urgency_lr.pkl")
URGENCY_VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "urgency", "urgency_tfidf_vectorizer.pkl")

# -------------------------------
# Load Models (Cached)
# -------------------------------
@st.cache_resource
def load_category_model():
    tokenizer = AutoTokenizer.from_pretrained(
        CATEGORY_MODEL_PATH, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CATEGORY_MODEL_PATH, local_files_only=True
    )
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_urgency_model():
    vectorizer = joblib.load(URGENCY_VECTORIZER_PATH)
    model = joblib.load(URGENCY_MODEL_PATH)
    return vectorizer, model

category_tokenizer, category_model = load_category_model()
urgency_vectorizer, urgency_model = load_urgency_model()

# -------------------------------
# Label Maps
# -------------------------------
CATEGORY_LABELS = {
    0: "complaint",
    1: "feedback",
    2: "other",
    3: "request",
    4: "spam"
}

URGENCY_LABELS = {0: "low", 1: "medium", 2: "high"}

# -------------------------------
# Rule-Based Logic
# -------------------------------
def rule_based_category(email_text, ml_label, confidence):
    text = email_text.lower()

    complaint_keywords = [
        "not working", "system down", "failed", "error",
        "issue", "problem", "crashed", "outage", "down"
    ]

    # Apply rule ONLY if confidence is low
    if confidence < 0.70:
        if any(k in text for k in complaint_keywords):
            return "complaint", "Rule-based override"

    return ml_label, "Model prediction"


def rule_based_urgency(email_text, ml_label, confidence):
    text = email_text.lower()

    high_urgency_keywords = [
        "urgent", "asap", "immediately", "critical",
        "right away", "priority", "emergency"
    ]

    if confidence < 0.70:
        if any(k in text for k in high_urgency_keywords):
            return "high", "Rule-based override"

    return ml_label, "Model prediction"

# -------------------------------
# Prediction Functions
# -------------------------------
def predict_category(email_text):
    inputs = category_tokenizer(
        email_text, return_tensors="pt",
        truncation=True, padding=True, max_length=256
    )

    with torch.no_grad():
        outputs = category_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    label = CATEGORY_LABELS[pred_class.item()]
    return label, confidence.item()


def predict_urgency(email_text):
    X = urgency_vectorizer.transform([email_text])
    pred = urgency_model.predict(X)[0]
    confidence = urgency_model.predict_proba(X).max()
    return pred, confidence

# -------------------------------
# UI
# -------------------------------
email_text = st.text_area(
    "✉️ Enter Email Text",
    height=200,
    placeholder="Example: Check for the DB errors asap."
)

if st.button("🔍 Predict"):
    if not email_text.strip():
        st.warning("Please enter an email.")
    else:
        with st.spinner("Analyzing email..."):
            # Category
            cat_label, cat_conf = predict_category(email_text)
            final_cat, cat_source = rule_based_category(
                email_text, cat_label, cat_conf
            )

            # Urgency
            urg_label, urg_conf = predict_urgency(email_text)
            final_urg, urg_source = rule_based_urgency(
                email_text, urg_label, urg_conf
            )

        st.success("Prediction Complete ✅")

        st.markdown("### 📂 Email Category")
        st.write(f"**{final_cat.capitalize()}**")
        st.caption(f"Confidence: {cat_conf:.2f} | Source: {cat_source}")

        st.markdown("### ⏱ Urgency Level")
        st.write(f"**{final_urg.capitalize()}**")
        st.caption(f"Confidence: {urg_conf:.2f} | Source: {urg_source}")

        st.markdown("---")
        st.markdown(
            "**Hybrid Decision Logic:**  \n"
            "- ML prediction used by default  \n"
            "- Rule-based override applied only when confidence is low"
        )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Milestone-4 | Hybrid ML + Rule-Based Streamlit Deployment")
