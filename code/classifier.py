import os
import pickle
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ACTION_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_action_model.pkl")
TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_type_model.pkl")

class LocalClassifier:
    def __init__(self):
        self.action_model = None
        self.type_model = None
        self.feature_names = []
        self._load_or_train_models()

    def _load_or_train_models(self):
        """Carrega ou treina o modelo de Regressão Logística leve."""
        if os.path.exists(ACTION_MODEL_PATH) and os.path.exists(TYPE_MODEL_PATH):
            try:
                with open(ACTION_MODEL_PATH, "rb") as f:
                    act_data = pickle.load(f)
                    self.action_model = act_data["model"]
                    self.feature_names = act_data["feature_names"]

                with open(TYPE_MODEL_PATH, "rb") as f:
                    typ_data = pickle.load(f)
                    self.type_model = typ_data["model"]

                return
            except Exception as e:
                pass

        # Train on startup if missing
        from train_ml_model import train_and_save_ml_model
        train_and_save_ml_model()

        if os.path.exists(ACTION_MODEL_PATH):
            with open(ACTION_MODEL_PATH, "rb") as f:
                act_data = pickle.load(f)
                self.action_model = act_data["model"]
                self.feature_names = act_data["feature_names"]

            with open(TYPE_MODEL_PATH, "rb") as f:
                typ_data = pickle.load(f)
                self.type_model = typ_data["model"]

    def classify(self, features: Dict[str, Any]) -> Tuple[str, str, float]:
        """Classifica as mensagens com o modelo de Machine Learning (Logistic Regression) leve."""
        
        # Build numeric feature vector matching trained feature set
        numeric_feats = {k: float(v) for k, v in features.items() if isinstance(v, (bool, int, float)) and k != "embedding"}
        
        if self.action_model and self.feature_names:
            vector = [numeric_feats.get(fn, 0.0) for fn in self.feature_names]
            X_input = pd.DataFrame([vector], columns=self.feature_names)

            # Predict Action & Probabilities
            probs = self.action_model.predict_proba(X_input)[0]
            winning_idx = probs.argmax()
            winning_action = str(self.action_model.classes_[winning_idx])
            confidence = float(probs[winning_idx])

            # Predict Message Type
            pred_type = str(self.type_model.predict(X_input)[0])

            return winning_action, pred_type, round(confidence, 2)

        # Emergency Fallback
        return "digest", "personal", 0.70
