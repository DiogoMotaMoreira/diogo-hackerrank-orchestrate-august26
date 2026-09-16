import os
import zipfile

def zip_folder():
    workspace_dir = "/home/diogo-moreira/Documents/GitHub/diogo-hackerrank-orchestrate-august26"
    zip_path = os.path.join(workspace_dir, "code.zip")
    
    exclude_dirs = {".git", "venv", "dataset"}
    exclude_files = {".env", "code.zip"}
    
    print(f"Creating submission zip: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(workspace_dir):
            # Prune directories we want to exclude
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                
                # Exclude media files to keep the zip lightweight
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, workspace_dir)
                
                if rel_path.startswith("dataset/media/"):
                    continue
                
                zipf.write(full_path, rel_path)
                print(f"  Added: {rel_path}")

    print(f"Successfully zipped submission package: {zip_path}")

if __name__ == "__main__":
    zip_folder()
