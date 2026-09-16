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
        
        # Default fallback
        winning_action = "digest"
        pred_type = "personal"
        confidence = 0.70

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

        # =========================================================================
        # ENSEMBLE DECISION FUSION & CONFIDENCE CALIBRATION
        # =========================================================================
        has_otp = numeric_feats.get("has_otp", 0.0) == 1.0
        is_phishing_link_scam = numeric_feats.get("is_phishing_link_scam", 0.0) == 1.0
        scam_risk = numeric_feats.get("scam_risk", 0.0) == 1.0
        during_quiet_hours = numeric_feats.get("during_quiet_hours", 0.0) == 1.0
        has_family = numeric_feats.get("has_family", 0.0) == 1.0
        has_health = numeric_feats.get("has_health", 0.0) == 1.0
        has_work = numeric_feats.get("has_work", 0.0) == 1.0
        group_is_muted_feat = numeric_feats.get("group_is_muted_feat", 0.0) == 1.0
        has_user_mention = numeric_feats.get("has_user_mention", 0.0) == 1.0
        group_muted_and_mention = numeric_feats.get("group_muted_and_mention", 0.0) == 1.0
        temporal_repetition_spam = numeric_feats.get("temporal_repetition_spam", 0.0) == 1.0
        message_similar_to_previous_spam = numeric_feats.get("message_similar_to_previous_spam", 0.0) == 1.0
        sender_trusted = numeric_feats.get("sender_trusted", 0.0) == 1.0

        # Rule 1: High-Priority OTP Bypass (Safe from Phishing)
        if has_otp and not is_phishing_link_scam:
            winning_action = "notify"
            confidence = max(confidence, 0.99)
            pred_type = "personal" if not numeric_feats.get("is_business", 0.0) else "business"

        # Rule 2: Phishing Scam / High Scam override
        elif is_phishing_link_scam or scam_risk or numeric_feats.get("has_scam", 0.0) == 1.0:
            winning_action = "mute"
            confidence = max(confidence, 0.98)

        # Rule 8: If user reported this sender before -> Mute
        elif numeric_feats.get("user_reported_sender_before", 0.0) == 1.0:
            winning_action = "mute"
            confidence = max(confidence, 0.97)

        # Rule 7: Peer-to-peer advertisements / sales -> Mute if user ignores similar or has high dismiss rate
        elif numeric_feats.get("is_peer_sale", 0.0) == 1.0:
            if message_similar_to_previous_spam == 1.0 or numeric_feats.get("hist_dismiss_rate", 0.0) > 0.5:
                winning_action = "mute"
                confidence = max(confidence, 0.90)
            else:
                winning_action = "digest"
                confidence = max(confidence, 0.85)

        # Rule 3: Muted Group Mentions (Upgrade to notify if mentioned)
        elif group_muted_and_mention:
            winning_action = "notify"
            confidence = max(confidence, 0.92)

        # Rule 4: Temporal Repetition / Similarity to past spam -> Mute
        elif (temporal_repetition_spam or message_similar_to_previous_spam) and not sender_trusted:
            winning_action = "mute"
            confidence = max(confidence, 0.95)

        # Rule 5: DND / Quiet Hours restriction (Downgrade to digest unless urgent)
        elif during_quiet_hours and not (has_family or has_health or has_work or has_otp):
            if winning_action == "notify":
                winning_action = "digest"
                confidence = max(confidence, 0.88)

        # Rule 6: Muted Group without mention -> Digest / Mute
        elif group_is_muted_feat and not has_user_mention:
            if winning_action == "notify":
                winning_action = "digest"
                confidence = max(confidence, 0.85)

        # Rule 9: Family message with health or call urgency -> Notify (unless it is casual / non-urgent)
        elif has_family and not numeric_feats.get("is_casual_non_urgent", 0.0) == 1.0 and (has_health or any(kw in str(features.get("full_text", "")).lower() for kw in ["call", "clinic", "unwell"])):
            winning_action = "notify"
            confidence = max(confidence, 0.94)

        return winning_action, pred_type, round(confidence, 2)
