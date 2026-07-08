"""Pre-download model to HF cache volume before starting vLLM server."""
import os
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Install huggingface-hub: pip install huggingface-hub")
    raise

model_id = os.getenv("MODEL_ID", "Qwen/Qwen3.6-35B-A3B")
cache_dir = os.getenv("HF_HOME", "/models")
print(f"Downloading {model_id} to {cache_dir}...")
snapshot_download(repo_id=model_id, cache_dir=cache_dir)
print("Download complete.")
