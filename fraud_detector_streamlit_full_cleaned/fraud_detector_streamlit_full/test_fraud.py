import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "fraud_detection_model.pkl"

model = joblib.load(MODEL_PATH)

samples = [
    "WINNER!! Claim your prize now",
    "Hello, how are you today?",
    "http://banking-update-login.cn/secure",
    "https://www.github.com"
]

for text in samples:
    pred = model.predict([text])[0]
    print(f"Input: {text} -> Prediction: {pred}")