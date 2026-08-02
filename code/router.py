import os
from typing import Dict, Any, List

try:
    from code.feature_extractor import FeatureExtractor
    from code.classifier import LocalClassifier
    from code.reason_generator import ReasonGenerator
except ImportError:
    from feature_extractor import FeatureExtractor
    from classifier import LocalClassifier
    from reason_generator import ReasonGenerator

ALLOWED_ACTIONS = ["notify", "digest", "mute"]
ALLOWED_MESSAGE_TYPES = [
    "personal", "urgent", "event", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown"
]

class MessageRouter:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.classifier = LocalClassifier()
        self.reason_generator = ReasonGenerator()
        print("[Router] Initialized 100% Local Offline Message Router (Zero-Cost ML Engine)")

    def route(self, message: Dict[str, Any], extracted_text: str, evidence: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Processa a mensagem e determina a ação, tipo de mensagem, razão e nível de confiança."""
        
        # 1. Extração de características (Features)
        features = self.feature_extractor.extract(message, extracted_text, context)

        # 2. Classificação local (Sem chamadas a APIs de LLMs)
        action, msg_type, confidence = self.classifier.classify(features)

        # Validação de tipos permitidos
        if action not in ALLOWED_ACTIONS:
            action = "digest"
        if msg_type not in ALLOWED_MESSAGE_TYPES:
            msg_type = "unknown"

        # 3. Geração de explicação determinística
        reason = self.reason_generator.generate_reason(action, msg_type, features)

        return {
            "action": action,
            "message_type": msg_type,
            "reason": reason,
            "confidence": round(confidence, 2)
        }