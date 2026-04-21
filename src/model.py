# src/model.py

import os
import io
import requests
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.preprocess import load_nslkdd, prepare_data, FEATURE_COLS

# NSL-KDD public download URL (20% sample for speed)
TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"
)
TEST_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
)

MODEL_PATH = "models/xgb_mitre.joblib"


def download_dataset(url: str) -> pd.DataFrame:
    from src.preprocess import NSL_KDD_COLUMNS
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), header=None, names=NSL_KDD_COLUMNS)
    return df


def train_model(df_train: pd.DataFrame):
    """Train XGBoost classifier and return (model, label_names, metrics_dict)."""
    X, y, label_names = prepare_data(df_train)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=label_names, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    metrics = {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "label_names": label_names,
        "feature_importances": model.feature_importances_,
        "feature_names": list(X.columns),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump((model, label_names), MODEL_PATH)

    return model, label_names, metrics


def load_or_train():
    """Load cached model or download data and train fresh."""
    if os.path.exists(MODEL_PATH):
        model, label_names = joblib.load(MODEL_PATH)
        return model, label_names, None

    df_train = download_dataset(TRAIN_URL)
    model, label_names, metrics = train_model(df_train)
    return model, label_names, metrics


def predict_single(model, label_names: list, feature_dict: dict) -> dict:
    """Predict MITRE category for a single sample dict."""
    from src.preprocess import encode_categoricals

    df = pd.DataFrame([feature_dict])
    df = encode_categoricals(df)
    X = df[FEATURE_COLS].fillna(0)

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_names[pred_idx]

    return {
        "predicted_category": pred_label,
        "confidence": float(proba[pred_idx]),
        "probabilities": {label_names[i]: float(proba[i]) for i in range(len(label_names))},
    }


def predict_batch(model, label_names: list, df: pd.DataFrame) -> pd.DataFrame:
    """Predict MITRE categories for a DataFrame of raw feature rows."""
    from src.preprocess import encode_categoricals

    df2 = encode_categoricals(df.copy())
    X = df2[FEATURE_COLS].fillna(0)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    df2["predicted_category"] = [label_names[p] for p in preds]
    df2["confidence"] = probas.max(axis=1)
    return df2
