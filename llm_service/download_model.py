"""Download GGUF model from HuggingFace Hub."""
from huggingface_hub import hf_hub_download
from .config import settings


def main() -> str:
    path = hf_hub_download(
        repo_id=settings.gguf_repo,
        filename=settings.gguf_filename,
        local_dir=settings.model_cache_dir,
    )
    print(f"Model downloaded to: {path}")
    return path


if __name__ == "__main__":
    main()
