import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# --- Page Setup ---
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")
st.title("📰 Fake News Headline Detector")
st.markdown("This app uses a machine learning model to predict whether a **news headline** is likely *Real* ✅ or *Fake* 🚨.")

st.markdown("---")

# Use session state to store input
if "example_text" not in st.session_state:
    st.session_state.example_text = ""

# Buttons that update the session state
col1, col2 = st.columns(2)
with col1:
    if st.button("Try Example (Fake)"):
        st.session_state.example_text = "Aliens have landed and taken over parliament 🚀"
with col2:
    if st.button("Try Example (Real)"):
        st.session_state.example_text = "Central bank raises interest rates by 0.25%"

# Input box (pre-filled from example)
user_input = st.text_area("✍️ Paste a news headline below:",
                          value=st.session_state.example_text,
                          placeholder="e.g. President signs new climate agreement")


# --- Prediction ---
if st.button("🧠 Predict"):
    if not user_input.strip():
        st.warning("Please enter a headline first.")
    else:
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max() * 100

        if prediction == 1:
            st.success(f"✅ **REAL NEWS** — Confidence: {confidence:.2f}%")
        else:
            st.error(f"🚨 **FAKE NEWS** — Confidence: {confidence:.2f}%")
