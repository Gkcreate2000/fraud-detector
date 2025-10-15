import os
import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SMS_DATA = os.path.join(DATA_DIR, "sms_spam.csv", "SMSSpamCollection")
URL_DATA = os.path.join(DATA_DIR, "url_phishing.csv", "urlset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detection_model.pkl")

def _read_sms(path: str) -> pd.DataFrame:
    """Read SMS spam dataset (tsv: label<tab>text) and map labels to 0/1."""
    df = pd.read_csv(path, sep='\t', header=None, names=['label','text'])
    df['label'] = df['label'].str.lower().map({'ham':0, 'spam':1}).astype(int)
    return df[['text','label']].dropna()

def _read_url(path: str) -> pd.DataFrame:
    """Read URL phishing dataset and map labels to 0/1 using the 'label' column.

    If no 'domain' column exists, try common alternatives.
    """
    # Try multiple encodings and parsing strategies
    for enc in ('utf-8','latin-1','cp1252'):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        # Fallback
        df = pd.read_csv(path, encoding='latin-1', engine='python', on_bad_lines='skip')

    # Column for URL text
    text_col = None
    for cand in ('domain','url','URL','link'):
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise ValueError("Could not find a URL/text column (looked for 'domain','url','URL','link').")

    # Column for label
    label_col = None
    for cand in ('label','Label','is_phishing','target'):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError("Could not find a label column (looked for 'label','is_phishing','target').")

    out = df[[text_col, label_col]].rename(columns={text_col:'text', label_col:'label'}).dropna()
    # Ensure binary 0/1
    if out['label'].dtype != int and out['label'].dtype != 'int64':
        # Map common representations
        mapping = {'phishing':1, 'legitimate':0, 'good':0, 'bad':1}
        out['label'] = out['label'].apply(lambda v: mapping.get(str(v).strip().lower(), v))
    out['label'] = out['label'].astype(float).astype(int)
    return out[['text','label']]

def _load_data() -> pd.DataFrame:
    sms_df = _read_sms(SMS_DATA)
    url_df = _read_url(URL_DATA)

    # Combine
    df = pd.concat([sms_df, url_df], ignore_index=True)
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != '']
    return df

def _train_model(df: pd.DataFrame):
    X = df['text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000)),
        ('logreg', LogisticRegression(max_iter=200, n_jobs=None))
    ])
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    return clf

def main():
    df = _load_data()
    os.makedirs(MODEL_DIR, exist_ok=True)
    model = _train_model(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()