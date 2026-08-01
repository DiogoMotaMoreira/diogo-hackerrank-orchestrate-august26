import os
import pandas as pd
from typing import Dict, Any, Tuple, List

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

        # Build composite index lookups
        self.group_member_lookup = {}
        for row in self.group_members_list:
            key = f"{row.get('group_id')}_{row.get('user_id')}"
            self.group_member_lookup[key] = row

        self.user_business_lookup = {}
        for row in self.user_business_history_list:
            key = f"{row.get('user_id')}_{row.get('business_id')}"
            self.user_business_lookup[key] = row

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

    def retrieve_evidence_and_context(self, message: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Finds historical evidence message IDs and builds enriched context features."""
        user_id = str(message.get("user_id", "") or "")
        group_id = str(message.get("group_id", "") or "")
        business_id = str(message.get("business_id", "") or "")
        sender_user_id = str(message.get("sender_user_id", "") or "")

        # Clean NaN strings
        if group_id == "nan": group_id = ""
        if business_id == "nan": business_id = ""
        if sender_user_id == "nan": sender_user_id = ""

        # Historical message evidence matching
        relevant_history = []
        for hist_msg in self.message_history:
            h_user = str(hist_msg.get("user_id", "") or "")
            h_group = str(hist_msg.get("group_id", "") or "")
            h_sender = str(hist_msg.get("sender_user_id", "") or hist_msg.get("business_id", "") or "")
            h_msg_id = str(hist_msg.get("message_id", "") or "")

            if h_user == user_id:
                if group_id and h_group == group_id:
                    relevant_history.append(h_msg_id)
                elif not group_id and (h_sender == sender_user_id or h_sender == business_id):
                    relevant_history.append(h_msg_id)

        evidence_str = ";".join(relevant_history[:3]) if relevant_history else "none"

        # Lookup composite user-group relationship
        group_member_info = self.group_member_lookup.get(f"{group_id}_{user_id}", {})
        group_meta = self.groups.get(group_id, {})
        if group_member_info:
            group_meta = {**group_meta, **group_member_info}
            # Set explicit flag for muted group
            group_meta["is_muted_by_user"] = bool(group_member_info.get("group_muted_by_user", 0) == 1)

        # Lookup composite user-business relationship
        user_bus_info = self.user_business_lookup.get(f"{user_id}_{business_id}", {})
        business_meta = self.business_accounts.get(business_id, {})
        if user_bus_info:
            business_meta = {**business_meta, **user_bus_info}

        # Calculate potential domain spoofing risk
        off_domain = str(business_meta.get("official_domain", "") or "").lower()
        sender_domain = str(business_meta.get("domain_used_by_sender", "") or "").lower()
        if off_domain and sender_domain and off_domain != sender_domain:
            business_meta["domain_mismatch_risk"] = True
        else:
            business_meta["domain_mismatch_risk"] = False

        context_features = {
            "user_pref": self.users.get(user_id, {}),
            "group_meta": group_meta,
            "business_meta": business_meta,
            "history_count": len(relevant_history)
        }

        return evidence_str, context_features