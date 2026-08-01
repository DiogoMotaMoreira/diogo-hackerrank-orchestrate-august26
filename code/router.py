import os
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

ALLOWED_ACTIONS = ["notify", "digest", "mute"]
ALLOWED_MESSAGE_TYPES = [
    "personal", "urgent", "event", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown"
]

class MessageRouter:
    def __init__(self):
        self.provider = None
        self.client = None
        self.model_name = None

        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if groq_key and groq_key.startswith("gsk_"):
            try:
                from groq import Groq
                self.client = Groq(api_key=groq_key)
                self.provider = "groq"
                self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                print(f"[Router] Initialized Groq LLM ({self.model_name})")
            except Exception as e:
                print(f"[Warning] Failed to initialize Groq: {e}")

        elif gemini_key and gemini_key.strip() and gemini_key != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.client = genai
                self.provider = "gemini"
                self.model_name = "gemini-1.5-flash"
                print(f"[Router] Initialized Gemini LLM ({self.model_name})")
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini: {e}")

    def route(self, message: Dict[str, Any], extracted_text: str, evidence: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Regras determinísticas de prioridade alta (Evita gastar chamadas em casos óbvios)
        hard_decision = self._check_hard_rules(message, extracted_text, context)
        if hard_decision:
            return hard_decision

        # 2. Roteamento por LLM focado na mensagem individual
        if self.provider:
            try:
                decision = self._route_with_llm(message, extracted_text, evidence, context)
                if decision:
                    return decision
            except Exception as e:
                print(f"[Warning] LLM Routing Error ({e}). Fallback to rules.")

        # 3. Fallback
        return self._route_with_rules(message, extracted_text, evidence, context)

    def _check_hard_rules(self, message: Dict[str, Any], extracted_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica regras estritas baseadas em metadados reais antes de consultar a IA."""
        group_meta = context.get("group_meta", {})
        user_pref = context.get("user_pref", {})
        business_meta = context.get("business_meta", {})
        
        raw_text = str(message.get("message_text", "") or message.get("text_content", "") or "").lower()
        full_text = f"{raw_text} {extracted_text}".lower().strip()

        # Grupo silenciado pelo utilizador -> MUTE
        if group_meta.get("is_muted_by_user") or group_meta.get("notification_setting") == "muted":
            return {
                "action": "mute",
                "message_type": "event" if "event" in full_text else "personal",
                "reason": "Message from an explicitly muted group.",
                "confidence": 0.98
            }

        # Spam Score elevado de empresas -> MUTE
        if business_meta.get("spam_score", 0) > 0.6 or "lottery" in full_text or "prize winner" in full_text:
            return {
                "action": "mute",
                "message_type": "scam" if "prize" in full_text or "winner" in full_text else "spam",
                "reason": "High spam/scam score detected in sender metadata.",
                "confidence": 0.95
            }

        # Códigos OTP / Verificação urgente -> NOTIFY
        if "otp" in full_text or "verification code" in full_text or "code is" in full_text:
            return {
                "action": "notify",
                "message_type": "urgent",
                "reason": "Time-sensitive OTP security verification code.",
                "confidence": 0.99
            }

        return None

    def _build_prompt(self, message: Dict[str, Any], extracted_text: str, evidence: str, context: Dict[str, Any]) -> str:
        raw_text = str(message.get("message_text", "") or message.get("text_content", "") or "")
        media_type = str(message.get("media_type", "") or "")
        forwarded_count = message.get("forwarded_count", 0)
        conversation_type = message.get("conversation_type", "")

        user_pref = context.get("user_pref", {})
        group_meta = context.get("group_meta", {})
        business_meta = context.get("business_meta", {})
        history_count = context.get("history_count", 0)

        return f"""You are a WhatsApp Notification Router AI. Your task is to accurately classify this message for a user.

INPUT DATA:
- Message Text: "{raw_text}"
- Media Type: {media_type}
- OCR / Transcribed Audio Text: "{extracted_text if extracted_text else 'None'}"
- Forwarded Count: {forwarded_count}
- Conversation Type: {conversation_type}
- Evidence Message IDs: {evidence}
- User Preferences: {json.dumps(user_pref)}
- Group Context: {json.dumps(group_meta)}
- Business Sender Metadata: {json.dumps(business_meta)}
- Interaction History Matches: {history_count}

CLASSIFICATION RULES:
1. "action":
   - "notify": OTPs, urgent personal alerts, direct mentions, immediate deadlines.
   - "digest": Routine non-urgent personal/work updates, scheduled events, transactional receipts, group chat messages.
   - "mute": Marketing, promotional, discounts, ads, spam, scam, forwarded chain messages, or messages from muted sources.

2. "message_type": MUST be EXACTLY one of: {json.dumps(ALLOWED_MESSAGE_TYPES)}
   - Use 'personal' for direct 1-on-1 human conversations.
   - Use 'greeting' for casual hellos/good morning messages.
   - Use 'promotion' for sales, discounts, or offers.
   - Use 'business_update' for transactional or account updates from companies.
   - Use 'event' for invitations or scheduled activities.
   - Use 'forward' if forwarded_count > 0 and it's a broadcast/chain message.

Output ONLY a JSON object:
{{
  "action": "<notify|digest|mute>",
  "message_type": "<one of allowed message_types>",
  "reason": "<short explanation>",
  "confidence": <float 0.0-1.0>
}}
"""

    def _route_with_llm(self, message: Dict[str, Any], extracted_text: str, evidence: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(message, extracted_text, evidence, context)

        try:
            if self.provider == "groq":
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                text = res.choices[0].message.content
                time.sleep(0.15)  # Pequeno atraso para respeitar os limites da Groq (30 RPM)
            elif self.provider == "gemini":
                model = self.client.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"response_mime_type": "application/json"}
                )
                res = model.generate_content(prompt)
                text = res.text
                time.sleep(1.2)

            return self._parse_llm_response(text)
        except Exception as e:
            print(f"[Warning] LLM Call Error ({e})")
            return {}

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        clean_text = text.strip()
        if clean_text.startswith("```json"): clean_text = clean_text[7:]
        if clean_text.startswith("```"): clean_text = clean_text[3:]
        if clean_text.endswith("```"): clean_text = clean_text[:-3]
        
        data = json.loads(clean_text.strip())
        action = str(data.get("action", "")).lower().strip()
        msg_type = str(data.get("message_type", "")).lower().strip()

        return {
            "action": action if action in ALLOWED_ACTIONS else "digest",
            "message_type": msg_type if msg_type in ALLOWED_MESSAGE_TYPES else "unknown",
            "reason": str(data.get("reason", "Decided by LLM Router.")).strip(),
            "confidence": min(max(float(data.get("confidence", 0.85)), 0.0), 1.0)
        }

    def _route_with_rules(self, message: Dict[str, Any], extracted_text: str, evidence: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": "digest",
            "message_type": "unknown",
            "reason": "Rule fallback.",
            "confidence": 0.50
        }