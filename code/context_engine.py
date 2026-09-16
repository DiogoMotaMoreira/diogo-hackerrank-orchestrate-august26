import os
import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime

def cosine_similarity(v1, v2):
    if not v1 or not v2: 
        return 0.0
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = sum(a*a for a in v1) ** 0.5
    norm2 = sum(b*b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0: 
        return 0.0
    return dot / (norm1 * norm2)

class ContextEngine:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.users = self._load_csv("users.csv", key_col="user_id")
        self.groups = self._load_csv("groups.csv", key_col="group_id")
        self.business_accounts = self._load_csv("business_accounts.csv", key_col="business_id")
        
        self.group_members_list = self._load_csv_list("group_members.csv")
        self.user_business_history_list = self._load_csv_list("user_business_history.csv")
        self.message_history = self._load_csv_list("message_history.csv")
        self.message_events = self._load_csv_list("message_events.csv")
        self.images = self._load_csv("images.csv", key_col="image_id")
        self.voice_notes = self._load_csv("voice_notes.csv", key_col="voice_note_id")

        # 1. Build composite index lookups
        self.group_member_lookup = {}
        for row in self.group_members_list:
            key = f"{row.get('group_id')}_{row.get('user_id')}"
            self.group_member_lookup[key] = row

        self.user_business_lookup = {}
        for row in self.user_business_history_list:
            key = f"{row.get('user_id')}_{row.get('business_id')}"
            self.user_business_lookup[key] = row

        # 2. Build index for message events
        self.events_by_msg_id = {}
        for row in self.message_events:
            msg_id = str(row.get("message_id", "")).strip()
            self.events_by_msg_id[msg_id] = row

        # 3. Build group membership role lookup
        self.group_role_lookup = {}
        for row in self.group_members_list:
            g_id = str(row.get("group_id", "")).strip()
            u_id = str(row.get("user_id", "")).strip()
            self.group_role_lookup[(g_id, u_id)] = row

        # 4. Group history by (user_id, sender_id), (user_id, group_id), and user_id
        self.history_by_recipient_sender = {}
        self.history_by_recipient_group = {}
        self.history_by_user = {}
        
        for row in self.message_history:
            u_id = str(row.get("user_id", "")).strip()
            g_id = str(row.get("group_id", "")).strip()
            if g_id == "nan": g_id = ""
            b_id = str(row.get("business_id", "")).strip()
            if b_id == "nan": b_id = ""
            s_id = str(row.get("sender_user_id", "")).strip()
            if s_id == "nan": s_id = ""
            
            sender_key = b_id if b_id else s_id
            
            if u_id not in self.history_by_user:
                self.history_by_user[u_id] = []
            self.history_by_user[u_id].append(row)
            
            if g_id:
                key = (u_id, g_id)
                if key not in self.history_by_recipient_group:
                    self.history_by_recipient_group[key] = []
                self.history_by_recipient_group[key].append(row)
            if sender_key:
                key = (u_id, sender_key)
                if key not in self.history_by_recipient_sender:
                    self.history_by_recipient_sender[key] = []
                self.history_by_recipient_sender[key].append(row)

        # 5. Build lookup for daily notifications summary
        self.daily_notifications = {}
        daily_list = self._load_csv_list("daily_notification_summary.csv")
        for row in daily_list:
            u_id = str(row.get("user_id", "")).strip()
            date_str = str(row.get("date", "")).strip()
            self.daily_notifications[(u_id, date_str)] = (
                int(row.get("notifications_sent", 0) or 0),
                int(row.get("notifications_dismissed", 0) or 0)
            )

        # 6. Load sentence transformer embedding model and pre-compute historical embeddings
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
            
        self.history_embeddings = {}
        if self.embedding_model and self.message_history:
            texts = [str(m.get("message_text", "")).strip() for m in self.message_history]
            msg_ids = [str(m.get("message_id", "")).strip() for m in self.message_history]
            try:
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
                for msg_id, emb in zip(msg_ids, embeddings):
                    self.history_embeddings[msg_id] = emb
            except Exception:
                pass

    def _load_csv(self, filename: str, key_col: str) -> Dict[str, Dict[str, Any]]:
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        return df.set_index(key_col).to_dict(orient="index")

    def _load_csv_list(self, filename: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            return []
        return pd.read_csv(path).to_dict(orient="records")

    def get_sender_type(self, sender_id: str):
        if sender_id in self.business_accounts:
            return 'business'
        return 'user'

    def _is_time_in_dnd(self, time_str: str, dnd_window: str) -> bool:
        if not dnd_window or "-" not in dnd_window:
            return False
        try:
            if " " in time_str:
                time_part = time_str.split(" ")[1]
            else:
                time_part = time_str
            
            h_msg, m_msg = map(int, time_part.split(":")[:2])
            t_msg = h_msg * 60 + m_msg
            
            start_str, end_str = dnd_window.split("-")
            h_start, m_start = map(int, start_str.split(":"))
            h_end, m_end = map(int, end_str.split(":"))
            
            t_start = h_start * 60 + m_start
            t_end = h_end * 60 + m_end
            
            if t_start <= t_end:
                return t_start <= t_msg <= t_end
            else:
                return t_msg >= t_start or t_msg <= t_end
        except Exception:
            return False

    def retrieve_evidence_and_context(self, message: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Finds historical evidence message IDs and builds enriched context features."""
        user_id = str(message.get("user_id", "") or "")
        group_id = str(message.get("group_id", "") or "")
        business_id = str(message.get("business_id", "") or "")
        sender_user_id = str(message.get("sender_user_id", "") or "")
        created_at = str(message.get("created_at", "") or "")
        msg_text = str(message.get("message_text", "") or "").strip().lower()

        if group_id == "nan": group_id = ""
        if business_id == "nan": business_id = ""
        if sender_user_id == "nan": sender_user_id = ""

        sender_id = business_id if business_id else sender_user_id

        # Retrieve relevant sender or group history
        hist_msgs = []
        if group_id:
            hist_msgs = self.history_by_recipient_group.get((user_id, group_id), [])
        elif sender_id:
            hist_msgs = self.history_by_recipient_sender.get((user_id, sender_id), [])

        # Calculate current message embedding
        current_emb = None
        if self.embedding_model and msg_text:
            try:
                current_emb = self.embedding_model.encode([msg_text], show_progress_bar=False)[0].tolist()
            except Exception:
                pass

        # 1. Cosine similarity-based retrieval and evidence ranking
        history_with_similarity = []
        for hm in hist_msgs:
            h_msg_id = str(hm.get("message_id", "")).strip()
            h_emb = self.history_embeddings.get(h_msg_id)
            
            sim = 0.0
            if current_emb and h_emb:
                sim = cosine_similarity(current_emb, h_emb)
            else:
                # Fallback to Jaccard / word overlap
                h_text = str(hm.get("message_text", "")).strip().lower()
                words1 = set(msg_text.split())
                words2 = set(h_text.split())
                if words1 or words2:
                    sim = len(words1 & words2) / len(words1 | words2)
            
            history_with_similarity.append((hm, sim))

        # Sort history by similarity descending
        history_with_similarity.sort(key=lambda x: x[1], reverse=True)

        # Retrieve Top-3 evidence message IDs with similarity >= 0.20
        evidence_list = []
        for hm, sim in history_with_similarity:
            if sim >= 0.20:
                evidence_list.append(str(hm.get("message_id", "")))
            if len(evidence_list) >= 3:
                break
        
        evidence_str = ";".join(evidence_list) if evidence_list else "none"

        # Lookup composite user-group relationship
        group_member_info = self.group_member_lookup.get(f"{group_id}_{user_id}", {})
        group_meta = self.groups.get(group_id, {})
        if group_member_info:
            group_meta = {**group_meta, **group_member_info}
            group_meta["is_muted_by_user"] = bool(group_member_info.get("group_muted_by_user", 0) == 1)

        # Lookup composite user-business relationship
        user_bus_info = self.user_business_lookup.get(f"{user_id}_{business_id}", {})
        business_meta = self.business_accounts.get(business_id, {})
        if user_bus_info:
            business_meta = {**business_meta, **user_bus_info}

        # Calculate potential domain spoofing risk
        off_domain = str(business_meta.get("official_domain", "") or "").lower()
        sender_domain = str(business_meta.get("domain_used_by_sender", "") or "").lower()
        domain_mismatch = bool(off_domain and sender_domain and off_domain != sender_domain)

        # Extract date and time parts
        msg_date = ""
        if " " in created_at:
            msg_date = created_at.split(" ")[0]

        # 2. Rich Behavioural and Contextual Features
        user_info = self.users.get(user_id, {})
        dnd_window = str(user_info.get("do_not_disturb_window", ""))
        during_quiet_hours = 1.0 if self._is_time_in_dnd(created_at, dnd_window) else 0.0

        is_verified_business = 1.0 if int(business_meta.get("verified", 0) or 0) == 1 else 0.0
        group_is_muted = 1.0 if (int(group_member_info.get("group_muted_by_user", 0) or 0) == 1 or group_meta.get("is_muted_by_user")) else 0.0
        
        sender_is_group_admin = 0.0
        if group_id and sender_user_id:
            sender_role_info = self.group_role_lookup.get((group_id, sender_user_id), {})
            sender_is_group_admin = 1.0 if sender_role_info.get("role") == "admin" else 0.0

        # Why user knows business
        why_knows = str(user_bus_info.get("why_user_knows_account", "")).lower()
        business_has_recent_order = 1.0 if any(term in why_knows for term in ["delivery", "order", "purchase", "grocery"]) else 0.0

        # Historical message open/reply/dismiss/report rates
        hist_count = float(len(hist_msgs))
        opened_count = 0
        replied_count = 0
        dismissed_count = 0
        reported_count = 0
        sender_hist_count = 0
        user_reported_sender_before = 0.0
        message_repeated_last_7_days = 0.0
        message_repeated_last_3_days = 0.0
        message_similar_to_previous_spam = 0.0

        try:
            current_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M")
        except Exception:
            current_dt = None

        # Check for semantic repetition/spam across the user's entire history (not just thread history)
        user_all_history = self.history_by_user.get(user_id, [])
        for hm in user_all_history:
            h_msg_id = str(hm.get("message_id", "")).strip()
            h_emb = self.history_embeddings.get(h_msg_id)
            sim = 0.0
            if current_emb and h_emb:
                sim = cosine_similarity(current_emb, h_emb)
            else:
                h_text = str(hm.get("message_text", "")).strip().lower()
                words1 = set(msg_text.split())
                words2 = set(h_text.split())
                if words1 or words2:
                    sim = len(words1 & words2) / len(words1 | words2)
            
            if sim >= 0.90 and current_dt:
                try:
                    h_dt = datetime.strptime(hm.get("created_at", ""), "%Y-%m-%d %H:%M")
                    diff_days = (current_dt - h_dt).days
                    if diff_days <= 7:
                        message_repeated_last_7_days = 1.0
                    if diff_days <= 3:
                        message_repeated_last_3_days = 1.0
                    
                    ev = self.events_by_msg_id.get(h_msg_id, {})
                    if ev.get("notification_dismissed", 0) == 1 or ev.get("muted_after_message", 0) == 1 or ev.get("message_reported", 0) == 1:
                        message_similar_to_previous_spam = 1.0
                except Exception:
                    pass

        for hm, sim in history_with_similarity:
            h_msg_id = str(hm.get("message_id", "")).strip()
            h_sender = hm.get("business_id", "") if hm.get("business_id", "") and str(hm.get("business_id", "")) != "nan" else hm.get("sender_user_id", "")
            if str(h_sender) == "nan": h_sender = ""
            
            ev = self.events_by_msg_id.get(h_msg_id, {})
            
            # Check if this historical message was from the same sender
            is_same_sender = bool(h_sender and str(h_sender) == str(sender_id))
            
            if is_same_sender:
                sender_hist_count += 1
                if ev.get("message_opened", 0) == 1:
                    opened_count += 1
                if ev.get("message_replied", 0) == 1:
                    replied_count += 1
                if ev.get("notification_dismissed", 0) == 1:
                    dismissed_count += 1
                if ev.get("message_reported", 0) == 1:
                    reported_count += 1
                    user_reported_sender_before = 1.0

        open_rate = opened_count / sender_hist_count if sender_hist_count > 0 else 0.0
        reply_rate = replied_count / sender_hist_count if sender_hist_count > 0 else 0.0
        dismiss_rate = dismissed_count / sender_hist_count if sender_hist_count > 0 else 0.0
        reported_rate = reported_count / sender_hist_count if sender_hist_count > 0 else 0.0

        user_replies_often_to_sender = 1.0 if (reply_rate > 0.25 and replied_count >= 1) or (int(user_bus_info.get("messages_replied_30d", 0) or 0) > 2) else 0.0
        sender_trusted = 1.0 if (is_verified_business == 1.0 or user_replies_often_to_sender == 1.0 or sender_hist_count > 10 or sender_is_group_admin == 1.0) else 0.0

        # Load daily stats
        daily_stats = self.daily_notifications.get((user_id, msg_date), (0, 0))
        notification_load_today = float(daily_stats[0])
        notification_dismissed_today = float(daily_stats[1])

        context_features = {
            "user_pref": user_info,
            "group_meta": group_meta,
            "business_meta": business_meta,
            "history_count": len(hist_msgs),
            
            # Flattened context features
            "is_verified_business": is_verified_business,
            "sender_trusted": sender_trusted,
            "user_replies_often_to_sender": user_replies_often_to_sender,
            "group_is_muted": group_is_muted,
            "sender_is_group_admin": sender_is_group_admin,
            "user_reported_sender_before": user_reported_sender_before,
            "business_has_recent_order": business_has_recent_order,
            "message_repeated_last_7_days": message_repeated_last_7_days,
            "message_repeated_last_3_days": message_repeated_last_3_days,
            "message_similar_to_previous_spam": message_similar_to_previous_spam,
            "during_quiet_hours": during_quiet_hours,
            "notification_load_today": notification_load_today,
            "notification_dismissed_today": notification_dismissed_today,
            "domain_mismatch": 1.0 if domain_mismatch else 0.0,
            "hist_open_rate": open_rate,
            "hist_reply_rate": reply_rate,
            "hist_dismiss_rate": dismiss_rate,
            "hist_reported_rate": reported_rate,
        }

        return evidence_str, context_features