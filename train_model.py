"""
train_model.py
--------------
Run this script locally to pre-train the XGBoost model and cache it
to models/xgb_mitre.joblib so Streamlit doesn't have to retrain on
every cold start.

Usage:
    python train_model.py
"""

import os
import sys
import time

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.model import download_dataset, train_model, TRAIN_URL, MODEL_PATH

if __name__ == "__main__":
    print("📥 Downloading NSL-KDD training data…")
    df = download_dataset(TRAIN_URL)
    print(f"   {len(df):,} rows loaded.")

    print("🤖 Training XGBoost model…")
    t0 = time.time()
    model, label_names, metrics = train_model(df)
    elapsed = time.time() - t0

    print(f"✅ Training complete in {elapsed:.1f}s")
    print(f"   Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"   Classes  : {label_names}")
    print(f"   Saved to : {MODEL_PATH}")
