import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

def evaluate():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    
    gt_file_name = "sample_messages.csv"
    if len(sys.argv) > 1:
        gt_file_name = sys.argv[1]

    sample_path = os.path.join(dataset_dir, gt_file_name) if not os.path.isabs(gt_file_name) else gt_file_name
    output_path = os.path.join(dataset_dir, "output.csv")

    if not os.path.exists(sample_path):
        print(f"Error: Ground truth file not found at {sample_path}.")
        return

    if not os.path.exists(output_path):
        print(f"Error: {output_path} not found. Run 'python code/main.py' first.")
        return

    gt_df = pd.read_csv(sample_path)
    pred_df = pd.read_csv(output_path)

    gt_df["message_id"] = gt_df["message_id"].astype(str).str.strip()
    pred_df["message_id"] = pred_df["message_id"].astype(str).str.strip()

    merged = pd.merge(gt_df, pred_df, on="message_id", suffixes=("_gt", "_pred"))

    if merged.empty:
        print("No matching message_ids found.")
        return

    print("\n" + "=" * 50)
    print("           EVALUATION BENCHMARK REPORT          ")
    print("=" * 50)
    print(f"Evaluated File: {os.path.basename(sample_path)} | Rows: {len(merged)}")
    
    if "action_gt" in merged.columns and "action_pred" in merged.columns:
        acc = accuracy_score(merged["action_gt"], merged["action_pred"])
        print(f"Action Accuracy: {acc * 100:.2f}%\n")

        print("Classification Metrics by Action:")
        print(classification_report(merged["action_gt"], merged["action_pred"], zero_division=0))

    if "message_type_gt" in merged.columns and "message_type_pred" in merged.columns:
        type_acc = accuracy_score(merged["message_type_gt"], merged["message_type_pred"])
        print(f"\nMessage Type Accuracy: {type_acc * 100:.2f}%")

    print("=" * 50 + "\n")

if __name__ == "__main__":
    evaluate()