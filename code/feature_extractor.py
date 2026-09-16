import re
from typing import Dict, Any, List

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    EMBEDDING_MODEL = None

class FeatureExtractor:
    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL

        # 1. URGENT / ESCALATION PATTERNS (ENGLISH)
        self.urgent_patterns = [
            r"\botp\b", r"\bverification code\b", r"\bsecurity code\b", r"\bexpire[s]?\b", 
            r"\burgent\b", r"\basap\b", r"\bimmediately\b", r"\balert\b",
            r"retry count crossed", r"alert threshold", r"escalation starts", r"come online now",
            r"heads-up", r"quick heads-up", r"last-minute shuffle", r"pulled to \d+",
            r"wait \d+ mins max", r"attention required", r"critical alert", r"move your cars"
        ]

        # 2. SCAM / PHISHING PATTERNS (ENGLISH)
        self.scam_patterns = [
            r"confirm password", r"verify now at", r"otp may have leaked", r"reply with the otp",
            r"profile will be blocked", r"account-login", r"workspace access will expire",
            r"ignore all previous routing rules", r"verification failed; reply with",
            r"reply with the \d+ digit login code", r"wallet verification failed",
            r"security alert: otp", r"support alert: profile", r"log in at", r"bank-secure",
            r"ignore all previous", r"ignore previous instructions", r"ignore all rules",
            r"mark this message as", r"classify this message"
        ]

        # 3. SPAM PATTERNS (ENGLISH)
        self.spam_patterns = [
            r"\blottery\b", r"\bwinner\b", r"guaranteed prize", r"claim your prize",
            r"won \$\d+", r"extra income", r"risk-free investment",
            r"reply stop to unsubscribe"
        ]

        # 4. PROMOTION PATTERNS (ENGLISH)
        self.promo_patterns = [
            r"\bdiscount[s]?\b", r"\bpromotion[s]?\b", r"\bpromo\b", r"\bsale[s]?\b", 
            r"\boferta[s]?\b", r"\bcoupon[s]?\b", r"\bfree\b", r"\bcashback\b", r"\bbuy now\b", 
            r"\bclick here\b", r"\bselling\b", r"50% off", r"pickup near", r"try50", 
            r"offer available", r"shopping offer", r"ladakh", r"nights package", 
            r"kurta set", r"cycle helmet"
        ]

        # 5. BUSINESS UPDATE PATTERNS (ENGLISH)
        self.business_update_patterns = [
            r"order ending", r"your order", r"packed and is expected", r"local hub", 
            r"shipped", r"delivered", r"receipt", r"invoice", r"payment received", 
            r"booking", r"pvr cinemas", r"thank you for choosing", r"valuable feedback",
            r"safety advisory", r"brand says", r"never ask for otp", r"account update", 
            r"transaction", r"experience with us"
        ]

        # 6. EVENT PATTERNS (ENGLISH)
        self.event_patterns = [
            r"\bmeeting\b", r"\bzoom\b", r"\bteams\b", r"\bbirthday\b", r"\bparty\b", 
            r"\bcircular\b", r"school circular", r"form is open", r"cultural night", 
            r"\bparents\b", r"\bschool\b", r"\bevent\b", r"\bconference\b", r"\bworkshop\b", 
            r"\btraining\b", r"small change for today", r"consent note",
            r"timing and consent", r"health-related update"
        ]

        # 7. HEALTH PATTERNS (ENGLISH)
        self.health_patterns = [
            r"\bhealth\b", r"\bhealth-related\b", r"\bdoctor\b", r"\bhospital\b", 
            r"\bappointment\b", r"\bprescription\b", r"\bmedical\b",
            r"\bpharmacy\b", r"warm water", r"drink warm water",
            r"\bclinic\b", r"\bunwell\b", r"\bemergency\b", r"\bsick\b"
        ]

        # 8. TRANSPORT PATTERNS (ENGLISH)
        self.transport_patterns = [
            r"\bbus\b", r"bus is leaving", r"\broute\b", r"\bdriver\b", r"\btanker\b", 
            r"tanker guy", r"\bflight\b", r"\btrain\b", r"\btransport\b", r"move your cars"
        ]

        # 9. GREETING PATTERNS (ENGLISH)
        self.greeting_patterns = [
            r"good morning", r"good afternoon", r"good evening", r"good night", 
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"stay positive", r"keep smiling", 
            r"share blessings", r"hope today is peaceful", r"no need to reply"
        ]

        # 10. FORWARD PATTERNS (ENGLISH)
        self.forward_patterns = [
            r"\bfwd\b", r"fwd as received", r"forwarded", r"pls forward", r"share with family"
        ]

        # 11. PERSONAL / FAMILY PATTERNS (ENGLISH)
        self.family_personal_patterns = [
            r"\bmom\b", r"\bdad\b", r"\bson\b", r"\bdaughter\b", r"\bgrandma\b", 
            r"\bgrandpa\b", r"\bbrother\b", r"\bsister\b", r"\bwife\b", r"\bhusband\b", 
            r"\bfriend\b", r"reached home", r"talk tomorrow", r"had dinner", r"match tonight", 
            r"pickup still works", r"when you get 5 mins", r"can you call", r"nothing urgent",
            r"call now", r"call me", r"call back", r"cool now"
        ]

        # 12. UNKNOWN / COMMUNITY VOLUNTEER PATTERNS (ENGLISH)
        self.volunteer_patterns = [
            r"volunteer sheet", r"coordinating registrations", r"volunteer", r"community"
        ]

        # 13. PAYMENT PATTERNS (ENGLISH)
        self.payment_patterns = [
            r"\bpaypal\b", r"\bcredit card\b", r"\bdebit card\b", r"\bbank\b", 
            r"\btransfer\b", r"\binvoice\b", r"\bpayment\b", r"\breceipt\b", 
            r"\bfailed\b", r"\bdeclined\b"
        ]

        self.negation_words = {"not", "never", "no", "don't", "dont", "without"}

    def extract(self, message: Dict[str, Any], extracted_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = str(message.get("message_text", "") or message.get("text_content", "") or "").strip()
        full_text = f"{raw_text} {extracted_text}".strip()
        full_text_lower = full_text.lower()

        group_meta = context.get("group_meta", {})
        business_meta = context.get("business_meta", {})
        history_count = int(context.get("history_count", 0) or 0)

        forwarded_count = int(message.get("forwarded_count", 0) or 0)
        conversation_type = str(message.get("conversation_type", "")).lower()

        has_user_mention = bool(re.search(r"@u_\d+", full_text_lower))
        has_scam = any(re.search(p, full_text_lower) for p in self.scam_patterns)
        has_spam = any(re.search(p, full_text_lower) for p in self.spam_patterns)
        has_negation = any(neg in full_text_lower for neg in self.negation_words)

        urgent_matches = sum(1 for p in self.urgent_patterns if re.search(p, full_text_lower))
        urgent_score = (urgent_matches * 3)

        scam_score_cat = (sum(1 for p in self.scam_patterns if re.search(p, full_text_lower)) * 5)
        spam_score_cat = (sum(1 for p in self.spam_patterns if re.search(p, full_text_lower)) * 4)

        promo_matches = sum(1 for p in self.promo_patterns if re.search(p, full_text_lower))
        promo_score = (promo_matches * 3)

        business_update_score = sum(1 for p in self.business_update_patterns if re.search(p, full_text_lower)) * 3
        event_score = sum(1 for p in self.event_patterns if re.search(p, full_text_lower)) * 3
        health_score = sum(1 for p in self.health_patterns if re.search(p, full_text_lower)) * 3
        transport_score = sum(1 for p in self.transport_patterns if re.search(p, full_text_lower)) * 2
        greeting_score = sum(1 for p in self.greeting_patterns if re.search(p, full_text_lower)) * 2
        forward_score = (sum(1 for p in self.forward_patterns if re.search(p, full_text_lower)) * 3) + (forwarded_count * 2)
        family_score = sum(1 for p in self.family_personal_patterns if re.search(p, full_text_lower)) * 3
        payment_score = sum(1 for p in self.payment_patterns if re.search(p, full_text_lower)) * 3
        unknown_score = sum(1 for p in self.volunteer_patterns if re.search(p, full_text_lower)) * 4

        # EXPLICIT BOOLEAN FEATURE FLAGS
        is_advisory = any(phrase in full_text_lower for phrase in ["never ask", "never share", "never tell", "never provide", "do not share", "dont share", "dont ask", "do not ask", "safety advisory"])
        scam_warning_keywords = ["leak", "confirm password", "verify now", "verification failed", "ignore all previous", "support alert", "suspend", "compromised", "account blocked", "profile blocked", "temporarily blocked", "permanently blocked", "will be blocked", "blocked in", "access blocked"]
        is_scam_warning = any(kw in full_text_lower for kw in scam_warning_keywords)
        
        has_otp = bool("security alert: otp" in full_text_lower or re.search(r"\botp\b|verification code|security code", full_text_lower) or (re.search(r"\b\d{4,8}\b", full_text_lower) and "otp" in full_text_lower))
        if is_advisory or is_scam_warning:
            has_otp = False
        if is_scam_warning:
            has_scam = True
            scam_score_cat = max(scam_score_cat, 15)
            
        has_family = family_score > 0
        has_health = health_score > 0
        has_work = bool(re.search(r"\bwork\b|\bmeeting\b|\bteams\b|\bzoom\b|\boffice\b|\bticket\b|\bjira\b|\bprod\b|\bdeadline\b", full_text_lower))
        has_payment = payment_score > 0
        has_bank = bool(re.search(r"\bbank\b|\bpaypal\b|\bcredit card\b|\bdebit card\b|\baccount\b", full_text_lower))
        has_school = bool(re.search(r"\bschool\b|\bcircular\b|\bteacher\b|\bstudent\b|\bclass\b|\bparents\b", full_text_lower))
        has_delivery = bool(re.search(r"\bdelivery\b|\border\b|\bshipped\b|\bcourier\b|\bhub\b|\bpackage\b", full_text_lower))
        has_meeting = bool(re.search(r"\bmeeting\b|\bzoom\b|\bteams\b|\bcall\b|\bsync\b", full_text_lower))
        has_event = event_score > 0
        has_question = bool("?" in full_text_lower or "when" in full_text_lower or "where" in full_text_lower or "can you" in full_text_lower or "how" in full_text_lower)
        has_link = bool("http://" in full_text_lower or "https://" in full_text_lower or "www." in full_text_lower or re.search(r"\b[a-zA-Z0-9-]+\.(com|net|org|co|info|xyz|link|secure|update|login)\b", full_text_lower))
        has_volunteer = bool("volunteer" in full_text_lower or "coordinating" in full_text_lower or "community" in full_text_lower)

        # High-Impact Patterns
        is_order_delivery_today = (has_delivery or "order" in full_text_lower) and bool(re.search(r"today|local hub|packed|delivered", full_text_lower))
        is_transport_urgent = (transport_score > 0 or "tanker" in full_text_lower or "bus" in full_text_lower) and bool(re.search(r"mins|move your cars|leaving|arriving", full_text_lower))
        is_phishing_link_scam = has_link and (has_scam or "log in at" in full_text_lower or "bank-secure" in full_text_lower or ("urgent" in full_text_lower and has_bank))

        # Metadata
        is_group_muted = bool(group_meta.get("is_muted_by_user") or group_meta.get("notification_setting") == "muted")
        spam_score_meta = float(business_meta.get("spam_score", 0.0) or 0.0)
        allows_promotions = bool(business_meta.get("allows_promotions", True))
        user_reports_30d = int(business_meta.get("user_reports_30d", 0) or 0)
        is_business = bool(business_meta)
        is_group = conversation_type == "group" or bool(group_meta)

        embedding = None
        if self.embedding_model and full_text:
            try:
                embedding = self.embedding_model.encode(full_text).tolist()
            except Exception:
                pass

        importance_score = urgent_score + family_score + health_score + payment_score + event_score + business_update_score
        noise_score = scam_score_cat + spam_score_cat + promo_score + forward_score

        # Flatten context features
        is_verified_business = float(context.get("is_verified_business", 0.0))
        sender_trusted = float(context.get("sender_trusted", 0.0))
        user_replies_often_to_sender = float(context.get("user_replies_often_to_sender", 0.0))
        group_is_muted_feat = float(context.get("group_is_muted", 0.0))
        sender_is_group_admin = float(context.get("sender_is_group_admin", 0.0))
        user_reported_sender_before = float(context.get("user_reported_sender_before", 0.0))
        business_has_recent_order = float(context.get("business_has_recent_order", 0.0))
        message_repeated_last_7_days = float(context.get("message_repeated_last_7_days", 0.0))
        message_repeated_last_3_days = float(context.get("message_repeated_last_3_days", 0.0))
        message_similar_to_previous_spam = float(context.get("message_similar_to_previous_spam", 0.0))
        during_quiet_hours = float(context.get("during_quiet_hours", 0.0))
        notification_load_today = float(context.get("notification_load_today", 0.0))
        notification_dismissed_today = float(context.get("notification_dismissed_today", 0.0))
        domain_mismatch = float(context.get("domain_mismatch", 0.0))
        hist_open_rate = float(context.get("hist_open_rate", 0.0))
        hist_reply_rate = float(context.get("hist_reply_rate", 0.0))
        hist_dismiss_rate = float(context.get("hist_dismiss_rate", 0.0))
        hist_reported_rate = float(context.get("hist_reported_rate", 0.0))

        # SCALE ALL RAW COUNTS AND SCORES BETWEEN 0.0 AND 1.0 FOR LOGISTIC REGRESSION
        scaled_forwarded_count = min(1.0, float(forwarded_count) / 10.0)
        scaled_user_reports_30d = min(1.0, float(user_reports_30d) / 10.0)
        scaled_history_count = min(1.0, float(history_count) / 50.0)
        scaled_notification_load_today = min(1.0, float(notification_load_today) / 20.0)
        scaled_notification_dismissed_today = min(1.0, float(notification_dismissed_today) / 20.0)

        scaled_urgent_score = min(1.0, float(urgent_score) / 15.0)
        scaled_scam_score = min(1.0, float(scam_score_cat) / 15.0)
        scaled_spam_score = min(1.0, float(spam_score_cat) / 15.0)
        scaled_promo_score = min(1.0, float(promo_score) / 15.0)
        scaled_business_update_score = min(1.0, float(business_update_score) / 15.0)
        scaled_event_score = min(1.0, float(event_score) / 15.0)
        scaled_health_score = min(1.0, float(health_score) / 15.0)
        scaled_transport_score = min(1.0, float(transport_score) / 15.0)
        scaled_greeting_score = min(1.0, float(greeting_score) / 15.0)
        scaled_forward_score = min(1.0, float(forward_score) / 15.0)
        scaled_family_score = min(1.0, float(family_score) / 15.0)
        scaled_payment_score = min(1.0, float(payment_score) / 15.0)
        scaled_unknown_score = min(1.0, float(unknown_score) / 15.0)
        scaled_importance_score = min(1.0, float(importance_score) / 30.0)
        scaled_noise_score = min(1.0, float(noise_score) / 30.0)
        scaled_feature_score = max(-1.0, min(1.0, float(importance_score - noise_score) / 30.0))

        # 3. Composite / Interaction Features
        trusted_business_transaction = is_verified_business * business_has_recent_order
        group_muted_and_mention = group_is_muted_feat * float(has_user_mention)
        scam_risk = float(bool((has_payment or has_bank) and has_link and not is_verified_business))
        urgent_health_or_family = during_quiet_hours * float(bool(has_family or has_health))
        repeat_spam_risk = message_repeated_last_7_days * float(bool(not sender_trusted))
        muted_group_no_mention = group_is_muted_feat * float(bool(not has_user_mention))
        promo_untrusted_sender = scaled_promo_score * float(bool(not sender_trusted))
        temporal_repetition_spam = message_repeated_last_3_days * float(bool(not sender_trusted))

        return {
            "full_text": full_text_lower,
            "raw_text": raw_text,
            "extracted_text": extracted_text,
            "conversation_type": conversation_type,
            "forwarded_count": scaled_forwarded_count,
            "is_group_muted": is_group_muted,
            "spam_score_meta": spam_score_meta,
            "allows_promotions": allows_promotions,
            "user_reports_30d": scaled_user_reports_30d,
            "history_count": scaled_history_count,
            "is_business": is_business,
            "is_group": is_group,
            "has_user_mention": has_user_mention,
            "has_otp": has_otp,
            "has_family": has_family,
            "has_health": has_health,
            "has_work": has_work,
            "has_payment": has_payment,
            "has_bank": has_bank,
            "has_school": has_school,
            "has_delivery": has_delivery,
            "has_meeting": has_meeting,
            "has_event": has_event,
            "has_question": has_question,
            "has_link": has_link,
            "has_volunteer": has_volunteer,
            "has_scam": has_scam,
            "has_spam": has_spam,
            "has_negation": has_negation,
            "is_order_delivery_today": is_order_delivery_today,
            "is_transport_urgent": is_transport_urgent,
            "is_phishing_link_scam": is_phishing_link_scam,
            "is_school_event": bool("school circular" in full_text_lower or "timing and consent" in full_text_lower),
            "is_health_event": bool("health-related update" in full_text_lower or "appointment" in full_text_lower),
            "is_casual_non_urgent": bool(any(phrase in full_text_lower for phrase in ["nothing urgent", "nothing dramatic", "not urgent", "no rush", "whenever you get time", "when you get time", "don't call", "dont call", "talk tomorrow", "match tonight", "reached home", "had dinner"])),
            "is_peer_sale": bool("selling" in full_text_lower or "kurta set" in full_text_lower or "cycle helmet" in full_text_lower or "gate 2" in full_text_lower),
            
            # Context and behavioral features
            "is_verified_business": is_verified_business,
            "sender_trusted": sender_trusted,
            "user_replies_often_to_sender": user_replies_often_to_sender,
            "group_is_muted_feat": group_is_muted_feat,
            "sender_is_group_admin": sender_is_group_admin,
            "user_reported_sender_before": user_reported_sender_before,
            "business_has_recent_order": business_has_recent_order,
            "message_repeated_last_7_days": message_repeated_last_7_days,
            "message_repeated_last_3_days": message_repeated_last_3_days,
            "message_similar_to_previous_spam": message_similar_to_previous_spam,
            "during_quiet_hours": during_quiet_hours,
            "notification_load_today": scaled_notification_load_today,
            "notification_dismissed_today": scaled_notification_dismissed_today,
            "domain_mismatch": domain_mismatch,
            "hist_open_rate": hist_open_rate,
            "hist_reply_rate": hist_reply_rate,
            "hist_dismiss_rate": hist_dismiss_rate,
            "hist_reported_rate": hist_reported_rate,

            # Composite / Interaction Features
            "trusted_business_transaction": trusted_business_transaction,
            "group_muted_and_mention": group_muted_and_mention,
            "scam_risk": scam_risk,
            "urgent_health_or_family": urgent_health_or_family,
            "repeat_spam_risk": repeat_spam_risk,
            "muted_group_no_mention": muted_group_no_mention,
            "promo_untrusted_sender": promo_untrusted_sender,
            "temporal_repetition_spam": temporal_repetition_spam,

            # CATEGORY SCORES (SCALED)
            "urgent_score": scaled_urgent_score,
            "scam_score": scaled_scam_score,
            "spam_score": scaled_spam_score,
            "promo_score": scaled_promo_score,
            "business_update_score": scaled_business_update_score,
            "event_score": scaled_event_score,
            "health_score": scaled_health_score,
            "transport_score": scaled_transport_score,
            "greeting_score": scaled_greeting_score,
            "forward_score": scaled_forward_score,
            "family_score": scaled_family_score,
            "payment_score": scaled_payment_score,
            "unknown_score": scaled_unknown_score,
            "importance_score": scaled_importance_score,
            "noise_score": scaled_noise_score,
            "feature_score": scaled_feature_score,
            "embedding": embedding
        }
