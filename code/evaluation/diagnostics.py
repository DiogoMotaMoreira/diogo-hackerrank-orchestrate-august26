import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_extractor import FeatureExtractor
from classifier import LocalClassifier
from context_engine import ContextEngine
from media_processor import MediaProcessor

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def run_diagnostics(dataset_filename: str = "sample_messages.csv"):
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    if not os.path.exists(dataset_path):
        dataset_path = dataset_filename

    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} not found.")
        return

    print("=" * 75)
    print(f"  DIAGNOSTICS & CROSS-VALIDATION BENCHMARK ({os.path.basename(dataset_path)})")
    print("=" * 75)

    df = pd.read_csv(dataset_path)
    extractor = FeatureExtractor()
    rule_classifier = LocalClassifier()
    context_eng = ContextEngine(dataset_dir)
    media_proc = MediaProcessor(dataset_dir)

    images_path = os.path.join(dataset_dir, "images.csv")
    voice_notes_path = os.path.join(dataset_dir, "voice_notes.csv")

    image_lookup = {}
    if os.path.exists(images_path):
        df_images = pd.read_csv(images_path)
        col_img = 'file_path' if 'file_path' in df_images.columns else ('path' if 'path' in df_images.columns else None)
        if 'image_id' in df_images.columns and col_img:
            image_lookup = df_images.set_index('image_id')[col_img].to_dict()

    voice_lookup = {}
    if os.path.exists(voice_notes_path):
        df_voice_notes = pd.read_csv(voice_notes_path)
        col_voice = 'file_path' if 'file_path' in df_voice_notes.columns else ('path' if 'path' in df_voice_notes.columns else None)
        if 'voice_note_id' in df_voice_notes.columns and col_voice:
            voice_lookup = df_voice_notes.set_index('voice_note_id')[col_voice].to_dict()

    extracted_records = []
    feature_activation_counts = {}

    for _, row in df.iterrows():
        msg_dict = row.to_dict()
        msg_type_raw = str(msg_dict.get("media_type", "text")).lower()
        media_path = str(msg_dict.get("media_path", "")).strip()
        media_id = str(msg_dict.get("media_id", "")).strip()
        extracted_text = ""
        if "image" in msg_type_raw:
            path_to_process = media_path if (media_path and 'media' in media_path) else image_lookup.get(media_id, "")
            extracted_text = media_proc.process_image(path_to_process or "")
        elif "voice" in msg_type_raw or "audio" in msg_type_raw:
            path_to_process = media_path if (media_path and 'media' in media_path) else voice_lookup.get(media_id, "")
            extracted_text = media_proc.process_voice_note(path_to_process or "")

        evidence_ids, ctx = context_eng.retrieve_evidence_and_context(msg_dict)
        feats = extractor.extract(msg_dict, extracted_text, ctx)

        for f_key, f_val in feats.items():
            if (isinstance(f_val, bool) and f_val) or (isinstance(f_val, (int, float)) and f_val == 1.0):
                feature_activation_counts[f_key] = feature_activation_counts.get(f_key, 0) + 1

        pred_action, pred_type, conf = rule_classifier.classify(feats)
        numeric_feats = {k: float(v) for k, v in feats.items() if isinstance(v, (bool, int, float)) and k != "embedding"}
        
        extracted_records.append({
            "message_id": str(row["message_id"]),
            "message_text": str(row.get("message_text", "")),
            "action_gt": str(row["action"]),
            "action_pred": pred_action,
            "message_type_gt": str(row["message_type"]),
            "message_type_pred": pred_type,
            "numeric_features": numeric_feats
        })

    eval_df = pd.DataFrame(extracted_records)
    labels = ["digest", "mute", "notify"]

    # 1. FEATURE ACTIVATION FREQUENCY
    print("\n1. FEATURE ACTIVATION FREQUENCY (Utilizações Reais):")
    print("-" * 55)
    print(f"{'Feature':<28} {'Utilizações':<15}")
    print("-" * 55)

    important_features = [
        "has_otp", "has_user_mention", "has_scam", "has_spam",
        "has_family", "has_health", "has_work", "has_payment",
        "has_school", "has_delivery", "has_meeting", "has_event",
        "has_bank", "has_question", "has_link", "has_volunteer",
        "is_casual_non_urgent", "is_order_delivery_today", 
        "is_transport_urgent", "is_phishing_link_scam",
        "is_verified_business", "sender_trusted", "group_is_muted_feat",
        "sender_is_group_admin", "during_quiet_hours"
    ]

    for feat in important_features:
        count = feature_activation_counts.get(feat, 0)
        print(f"{feat:<28} {count:<15}")

    # 2. CONFUSION MATRIX (Rule Router Actions)
    print("\n2. CONFUSION MATRIX (Pure Rule Accumulator Actions):")
    print("-" * 55)
    cm = confusion_matrix(eval_df["action_gt"], eval_df["action_pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df.to_string())

    # 2.2 CONFUSION MATRIX (Message Types)
    print("\n2.2 CONFUSION MATRIX (Message Types):")
    print("-" * 55)
    unique_types = sorted(list(set(eval_df["message_type_gt"].unique()) | set(eval_df["message_type_pred"].unique())))
    cm_types = confusion_matrix(eval_df["message_type_gt"], eval_df["message_type_pred"], labels=unique_types)
    cm_types_df = pd.DataFrame(cm_types, index=[f"Actual {t}" for t in unique_types], columns=[f"Pred {t}" for t in unique_types])
    print(cm_types_df.to_string())

    # 2.3 MESSAGE TYPE ACCURACY BREAKDOWN
    print("\n2.3 MESSAGE TYPE ACCURACY BREAKDOWN:")
    print("-" * 55)
    for mt in unique_types:
        total_gt = (eval_df["message_type_gt"] == mt).sum()
        correct = ((eval_df["message_type_gt"] == mt) & (eval_df["message_type_pred"] == mt)).sum()
        pct = (correct / total_gt * 100.0) if total_gt > 0 else 0.0
        print(f"Type: {mt:<18} | Correct: {correct:>2} / {total_gt:<2} | Accuracy: {pct:.2f}%")

    # 3. DETAILED MISS ANALYSIS & ERROR PATTERNS
    print("\n3. DETAILED MISS ANALYSIS & ERROR PATTERNS:")
    print("-" * 55)

    # Notify predicted as Digest
    notify_as_digest = eval_df[(eval_df["action_gt"] == "notify") & (eval_df["action_pred"] == "digest")]
    print(f"\n[A] NOTIFY classified as DIGEST ({len(notify_as_digest)} cases):")
    if notify_as_digest.empty:
        print("    -> None! Perfect 0 miss rate.")
    else:
        for _, r in notify_as_digest.iterrows():
            print(f"    - ID: {r['message_id']} | Text: \"{r['message_text'][:70]}...\"")

    # Digest predicted as Mute
    digest_as_mute = eval_df[(eval_df["action_gt"] == "digest") & (eval_df["action_pred"] == "mute")]
    print(f"\n[B] DIGEST classified as MUTE ({len(digest_as_mute)} cases):")
    if digest_as_mute.empty:
        print("    -> None! Perfect 0 miss rate.")
    else:
        for _, r in digest_as_mute.iterrows():
            print(f"    - ID: {r['message_id']} | Text: \"{r['message_text'][:70]}...\"")

    # Total Errors Breakdown
    other_errors = eval_df[eval_df["action_gt"] != eval_df["action_pred"]]
    print(f"\n[C] TOTAL ERRORS BREAKDOWN ({len(other_errors)} / {len(eval_df)}):")
    if other_errors.empty:
        print("    -> 100% Perfect classification across all messages!")
    else:
        for _, r in other_errors.iterrows():
            print(f"    - ID: {r['message_id']} | GT: {r['action_gt'].upper()} -> PRED: {r['action_pred'].upper()} | Text: \"{r['message_text'][:60]}...\"")

    # 4. MACHINE LEARNING EVALUATION (5-FOLD CROSS VALIDATION & TRAIN/TEST SPLIT)
    print("\n4. MACHINE LEARNING EVALUATION (Preventing Overfitting with Cross-Validation):")
    print("=" * 75)

    X_df = pd.DataFrame([r["numeric_features"] for r in extracted_records]).fillna(0.0)
    y_action = eval_df["action_gt"]

    rule_acc = accuracy_score(eval_df["action_gt"], eval_df["action_pred"])

    # 5-Fold Stratified Cross-Validation for Logistic Regression & Random Forest
    n_samples = len(df)
    n_splits = min(5, max(2, n_samples // 3))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # A. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr_cv_scores = cross_val_score(lr, X_df, y_action, cv=skf)
    
    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_action, test_size=0.2, random_state=42, stratify=y_action)
    lr.fit(X_train, y_train)
    lr_test_acc = accuracy_score(y_test, lr.predict(X_test))

    # B. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cv_scores = cross_val_score(rf, X_df, y_action, cv=skf)
    rf.fit(X_train, y_train)
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

    # C. XGBoost
    xgb_cv_mean_str = "N/A"
    xgb_test_acc_str = "N/A"

    if HAS_XGB:
        label_map = {"digest": 0, "mute": 1, "notify": 2}
        y_int = y_action.map(label_map)
        y_train_int = y_train.map(label_map)
        y_test_int = y_test.map(label_map)

        xgb = XGBClassifier(random_state=42, eval_metric="mlogloss")
        xgb_cv_scores = cross_val_score(xgb, X_df, y_int, cv=skf)
        xgb_cv_mean_str = f"{xgb_cv_scores.mean() * 100:.2f}% (±{xgb_cv_scores.std() * 100:.2f}%)"

        xgb.fit(X_train, y_train_int)
        xgb_test_acc = accuracy_score(y_test_int, xgb.predict(X_test))
        xgb_test_acc_str = f"{xgb_test_acc * 100:.2f}%"

    print(f"{'Model Architecture':<30} {'5-Fold CV Accuracy':<25} {'80/20 Test Set Acc':<20}")
    print("-" * 75)
    print(f"{'1. Pure Rule Accumulator':<30} {rule_acc * 100:.2f}% (Deterministic)    {rule_acc * 100:.2f}%")
    print(f"{'2. Logistic Regression':<30} {lr_cv_scores.mean() * 100:.2f}% (±{lr_cv_scores.std() * 100:.2f}%)   {lr_test_acc * 100:.2f}%")
    print(f"{'3. Random Forest Classifier':<30} {rf_cv_scores.mean() * 100:.2f}% (±{rf_cv_scores.std() * 100:.2f}%)   {rf_test_acc * 100:.2f}%")
    print(f"{'4. XGBoost Classifier':<30} {xgb_cv_mean_str:<25} {xgb_test_acc_str}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv) > 1 else "sample_messages.csv"
    run_diagnostics(fn)
