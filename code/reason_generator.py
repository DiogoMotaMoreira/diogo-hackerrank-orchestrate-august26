from typing import Dict, Any

class ReasonGenerator:
    def generate_reason(self, action: str, message_type: str, features: Dict[str, Any]) -> str:
        """Gera uma explicação humana e precisa para a decisão tomada com base nos fatores acumulados."""
        
        scam_score = features.get("scam_score", 0)
        family_score = features.get("family_score", 0)
        health_score = features.get("health_score", 0)
        urgent_score = features.get("urgent_score", 0)
        history_count = features.get("history_count", 0)
        is_payment_urgent = features.get("is_payment_urgent", False)

        if action == "mute":
            if scam_score > 0 or features.get("has_scam"):
                return "Phishing scam attempt requesting password verification or fake OTP access."
            if features.get("is_group_muted"):
                return "Message from an explicitly muted group according to user preferences."
            if features.get("spam_score_meta", 0) > 0.4:
                return f"High spam risk detected from sender metadata (spam_score: {features['spam_score_meta']:.2f})."
            if features.get("user_reports_30d", 0) > 3:
                return f"Sender has multiple recent spam reports ({features['user_reports_30d']} reports in 30 days)."
            if not features.get("allows_promotions", True):
                return "User has opted out of promotional messages from this business sender."
            if features.get("promo_score", 0) > 0:
                return "Contains promotional marketing offers or discount deals."
            if features.get("forwarded_count", 0) >= 3:
                return f"Broadcast chain message forwarded {features['forwarded_count']} times."
            return "Suppressed as low-value or promotional notification."

        elif action == "notify":
            if is_payment_urgent:
                return "Urgent financial alert or failed transaction notification requiring immediate action."
            if family_score > 0 and health_score > 0:
                return "Critical health or medical update from family contact."
            if features.get("has_user_mention"):
                return "Direct user mention requiring immediate attention."
            if features.get("has_otp"):
                return "Time-sensitive security verification code (OTP) detected."
            if urgent_score > 0:
                return "Urgent time-sensitive alert requiring immediate user attention."
            if family_score > 0:
                return "Important message from direct family contact."
            if health_score > 0:
                return "Health or medical appointment notification."
            if history_count > 100:
                return f"High priority direct message from very frequent contact ({history_count} past interactions)."
            return "High priority direct notification."

        else: # digest
            if features.get("event_score", 0) > 0:
                return "Scheduled calendar event or meeting invitation aggregated for digest."
            if message_type == "greeting":
                return "Casual greeting summarized for daily notification digest."
            if message_type == "business_update":
                return "Non-urgent business update or transactional receipt."
            if message_type == "promotion":
                return "Promotional offer saved for scheduled digest summary."
            if features.get("is_group"):
                return "Routine group chat message aggregated for scheduled digest delivery."
            if features.get("forwarded_count", 0) > 0:
                return "Forwarded media message saved for digest summary."
            return "Standard message saved for scheduled notification digest."
