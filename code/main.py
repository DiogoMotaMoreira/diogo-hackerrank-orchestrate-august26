import os
import sys
import pandas as pd
from media_processor import MediaProcessor
from context_engine import ContextEngine
from router import MessageRouter

def log_transcript(entry: str):
    home_dir = os.path.expanduser("~")
    log_dir = os.path.join(home_dir, "hackerrank_orchestrate_august26")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    input_filename = sys.argv[1] if len(sys.argv) > 1 else "messages.csv"
    messages_path = os.path.join(dataset_dir, input_filename)
    output_path = os.path.join(dataset_dir, "output.csv")
    images_path = os.path.join(dataset_dir, "images.csv")
    voice_notes_path = os.path.join(dataset_dir, "voice_notes.csv")

    print(f"Loading dataset from: {dataset_dir}")
    print(f"Processing input file: {input_filename}")
    if not os.path.exists(messages_path):
        print(f"Error: {messages_path} not found.")
        sys.exit(1)

    df_messages = pd.read_csv(messages_path)

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

    print(f"Processing {len(df_messages)} input messages...")

    media_proc = MediaProcessor(dataset_dir)
    context_eng = ContextEngine(dataset_dir)
    router = MessageRouter()

    results = []

    for idx, row in df_messages.iterrows():
        msg_dict = row.to_dict()
        msg_id = str(msg_dict.get("message_id", "")).strip()
        msg_type_raw = str(msg_dict.get("media_type", "text")).lower()

        extracted_text = ""
        media_path = str(msg_dict.get("media_path", "")).strip()
        media_id = str(msg_dict.get("media_id", "")).strip()

        if "image" in msg_type_raw:
            path_to_process = media_path if (media_path and 'media' in media_path) else image_lookup.get(media_id, "")
            extracted_text = media_proc.process_image(path_to_process or "")
        elif "voice" in msg_type_raw or "audio" in msg_type_raw:
            path_to_process = media_path if (media_path and 'media' in media_path) else voice_lookup.get(media_id, "")
            extracted_text = media_proc.process_voice_note(path_to_process or "")

        evidence_ids, ctx_features = context_eng.retrieve_evidence_and_context(msg_dict)
        decision = router.route(msg_dict, extracted_text, evidence_ids, ctx_features)

        results.append({
            "message_id": msg_id,
            "action": decision["action"],
            "message_type": decision["message_type"],
            "reason": decision["reason"],
            "confidence": round(float(decision["confidence"]), 4),
            "evidence_message_ids": evidence_ids
        })

    out_df = pd.DataFrame(results, columns=[
        "message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"
    ])
    out_df.to_csv(output_path, index=False)
    print(f"Predictions successfully generated -> {output_path}")

    log_entry = f"[Execution Summary] Processed {len(df_messages)} messages from {input_filename}."
    log_transcript(log_entry)

if __name__ == "__main__":
    main()