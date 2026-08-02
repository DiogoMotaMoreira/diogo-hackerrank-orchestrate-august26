import os
import sys
import pickle
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_extractor import FeatureExtractor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ACTION_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_action_model.pkl")
TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_type_model.pkl")

def train_and_save_ml_model():
    print("Training lightweight Logistic Regression Classifier for Mobile WhatsApp Router...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "sample_messages.csv"))
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found!")
        return

    df = pd.read_csv(dataset_path)
    extractor = FeatureExtractor()

    X_list = []
    y_action = []
    y_type = []

    for _, row in df.iterrows():
        msg_dict = row.to_dict()
        ctx = {"group_meta": {}, "business_meta": {}, "history_count": 0}
        feats = extractor.extract(msg_dict, str(row.get("message_text", "")), ctx)
        numeric_feats = {k: float(v) for k, v in feats.items() if isinstance(v, (bool, int, float)) and k != "embedding"}
        
        X_list.append(numeric_feats)
        y_action.append(str(row["action"]))
        y_type.append(str(row["message_type"]))

    X_df = pd.DataFrame(X_list).fillna(0.0)
    feature_names = list(X_df.columns)

    # Train Logistic Regression Models
    action_model = LogisticRegression(max_iter=1000, random_state=42, C=1.5)
    action_model.fit(X_df, y_action)

    type_model = LogisticRegression(max_iter=1000, random_state=42, C=1.5)
    type_model.fit(X_df, y_type)

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(ACTION_MODEL_PATH, "wb") as f:
        pickle.dump({"model": action_model, "feature_names": feature_names}, f)

    with open(TYPE_MODEL_PATH, "wb") as f:
        pickle.dump({"model": type_model, "feature_names": feature_names}, f)

    print(f"Models successfully trained & persisted to {MODEL_DIR}!")

if __name__ == "__main__":
    train_and_save_ml_model()
