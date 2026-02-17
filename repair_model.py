import urllib.request
import os

def repair_model():
    base_url = "https://huggingface.co/patrickjohncyh/fashion-clip/raw/main/"
    files = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    dest_dir = "models/fashion-clip"

    print(f"Creating directory: {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    for f in files:
        dest_path = os.path.join(dest_dir, f)
        if os.path.exists(dest_path):
            print(f"[REPAIR] {f} already exists, skipping.")
            continue
        print(f"[REPAIR] Downloading {f}...")
        try:
            urllib.request.urlretrieve(base_url + f, dest_path)
        except Exception as e:
            print(f"[REPAIR] Error downloading {f}: {e}")

    # Handle the big weights file
    source_weights = "models/fashion-clip.bin"
    target_weights = os.path.join(dest_dir, "pytorch_model.bin")
    
    if os.path.exists(source_weights):
        print(f"[REPAIR] Found weights at {source_weights}. Moving to {target_weights}...")
        if os.path.exists(target_weights):
            os.remove(target_weights)
        os.rename(source_weights, target_weights)
        print("[REPAIR] Weights moved successfully.")
    elif os.path.exists(target_weights):
        print("[REPAIR] Weights already in correct location.")
    else:
        print("[REPAIR] WARNING: Could not find 'models/fashion-clip.bin'. Please ensure it exists.")

if __name__ == "__main__":
    repair_model()
