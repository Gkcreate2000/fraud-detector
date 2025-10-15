import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import time

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
(MODEL_PATH.parent).mkdir(exist_ok=True)

# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(
    page_title="🚨 Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# ----------------- THEMES -----------------
THEMES = {
    "Dark": """
        <style>
        .stApp { 
            background-color: #0f1117; 
        }
        .main-header, .subheader, .stRadio, .stFileUploader, .stButton, .stTextInput, .stProgress, .stCaption {
            color: #fafafa !important;
        }
        .stTextInput input {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #4a4a4a;
        }
        .stSidebar {
            background-color: #0f1117;
        }
        </style>
    """,
    "RGB Neon": """
        <style>
        @keyframes rgb {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .stApp {
          background: linear-gradient(270deg, #ff0080, #7928ca, #2afeb7);
          background-size: 600% 600%;
          animation: rgb 12s ease infinite;
        }
        .main-header, .subheader, .stRadio, .stFileUploader, .stButton, .stTextInput, .stProgress, .stCaption {
            color: #ffffff !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }
        .stTextInput input {
            background-color: rgba(255,255,255,0.9);
            color: #000000;
            border: 1px solid #ffffff;
        }
        .css-1d391kg, .css-12oz5g7 {
            background-color: transparent;
        }
        .stSidebar {
            background-color: rgba(15, 17, 23, 0.8);
        }
        </style>
    """
}

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        # Fallback simple model
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=200))
        ])
        # Train on tiny seed dataset
        X = ["Congratulations! You won", "Hello friend how are you", "Urgent! Verify account at link", "Your Amazon order shipped"]
        y = [1, 0, 1, 0]
        pipe.fit(X, y)
        joblib.dump(pipe, MODEL_PATH)
        return pipe

model = load_model()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="subheader">⚙️ Settings</div>', unsafe_allow_html=True)
    theme_choice = st.radio("Choose Theme", list(THEMES.keys()))
    st.markdown(THEMES[theme_choice], unsafe_allow_html=True)

    st.markdown('<div class="subheader">📂 Upload CSV to Train</div>', unsafe_allow_html=True)
    st.caption("CSV must have two columns: `text`, `label` (1=fraud/spam, 0=legit)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if {"text", "label"}.issubset(df.columns):
            st.success(f"Loaded {len(df)} rows. Training model...")
            X_train, X_test, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=0.2, random_state=42
            )
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression(max_iter=500))
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            joblib.dump(pipe, MODEL_PATH)
            df.to_csv(DATA_DIR / f"dataset_{int(time.time())}.csv", index=False)
            st.success(f"✅ Model trained & saved. Accuracy: {acc:.2%}")
            model = pipe
        else:
            st.error("CSV must contain `text` and `label` columns.")

# ----------------- MAIN -----------------
st.markdown('<div class="main-header">🚨 Unified Fraud / Spam / Phishing Detector</div>', unsafe_allow_html=True)
st.markdown("Enter an SMS, message, or URL to check if it's suspicious.")

user_input = st.text_input("🔎 Enter Text or URL", placeholder="e.g., WINNER! Claim reward at http://bit.ly/…")

if st.button("Predict"):
    if user_input.strip():
        pred = model.predict([user_input.strip()])[0]
        proba = getattr(model, "predict_proba", lambda X: [[0.0, 0.0]])([user_input.strip()])[0]
        label = "⚠️ Suspicious" if pred == 1 else "✅ Looks Legitimate"
        st.subheader(label)
        if len(proba) == 2:
            # Use appropriate progress bar color based on theme
            if theme_choice == "Dark":
                st.progress(float(proba[1]))
            else:
                # For RGB Neon theme, create a custom progress bar
                progress_html = f"""
                <div style="background-color: rgba(0,0,0,0.3); border-radius: 10px; padding: 3px;">
                    <div style="background-color: {'#ff4b4b' if pred == 1 else '#00cc66'}; width: {proba[1]*100}%; 
                                height: 20px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {proba[1]:.1%}
                    </div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
            
            st.caption(f"Confidence → Suspicious: {proba[1]:.2%} | Legit: {proba[0]:.2%}")
    else:
        st.warning("Please enter some text or URL.")

st.markdown("---")
st.caption("🔐 Built with Streamlit, scikit-learn, and ❤️ by GK")