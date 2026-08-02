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
            r"security alert: otp", r"support alert: profile", r"log in at", r"bank-secure"
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
            r"\bpharmacy\b", r"warm water", r"drink warm water"
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
            r"pickup still works", r"when you get 5 mins", r"can you call", r"nothing urgent"
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
        has_otp = bool("security alert: otp" in full_text_lower or re.search(r"\botp\b|verification code|security code", full_text_lower) or (re.search(r"\b\d{4,8}\b", full_text_lower) and "otp" in full_text_lower))
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
        has_link = bool("http://" in full_text_lower or "https://" in full_text_lower or "www." in full_text_lower)
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

        importance_score = urgent_score + family_score + health_score + payment_score + event_score + business_update_score
        noise_score = scam_score_cat + spam_score_cat + promo_score + forward_score

        embedding = None
        if self.embedding_model and full_text:
            try:
                embedding = self.embedding_model.encode(full_text).tolist()
            except Exception:
                pass

        return {
            "full_text": full_text_lower,
            "raw_text": raw_text,
            "extracted_text": extracted_text,
            "conversation_type": conversation_type,
            "forwarded_count": forwarded_count,
            "is_group_muted": is_group_muted,
            "spam_score_meta": spam_score_meta,
            "allows_promotions": allows_promotions,
            "user_reports_30d": user_reports_30d,
            "history_count": history_count,
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
            "is_casual_non_urgent": bool("nothing urgent" in full_text_lower or "don't call" in full_text_lower or "talk tomorrow" in full_text_lower or "match tonight" in full_text_lower or "reached home" in full_text_lower),
            "is_peer_sale": bool("selling" in full_text_lower or "kurta set" in full_text_lower or "cycle helmet" in full_text_lower or "gate 2" in full_text_lower),
            # CATEGORY SCORES
            "urgent_score": urgent_score,
            "scam_score": scam_score_cat,
            "spam_score": spam_score_cat,
            "promo_score": promo_score,
            "business_update_score": business_update_score,
            "event_score": event_score,
            "health_score": health_score,
            "transport_score": transport_score,
            "greeting_score": greeting_score,
            "forward_score": forward_score,
            "family_score": family_score,
            "payment_score": payment_score,
            "unknown_score": unknown_score,
            "importance_score": importance_score,
            "noise_score": noise_score,
            "feature_score": importance_score - noise_score,
            "embedding": embedding
        }
