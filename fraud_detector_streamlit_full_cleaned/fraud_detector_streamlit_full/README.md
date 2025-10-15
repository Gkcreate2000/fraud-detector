# Fraud Detector (Spam + Phishing)

A simple pipeline that trains a TF-IDF + Logistic Regression model using:
- **SMS Spam** dataset (`data/sms_spam.csv/SMSSpamCollection`)
- **URL Phishing** dataset (`data/url_phishing.csv/urlset.csv`)

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

This will save a model to `models/fraud_detection_model.pkl`.

## Run Streamlit App

```bash
streamlit run app_streamlit.py
```

## Quick Test

```bash
python test_fraud.py
```