from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    gguf_repo: str = "bartowski/Llama-3.2-1B-Instruct-GGUF"
    gguf_filename: str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    model_cache_dir: str = str(Path.home() / ".cache" / "huggingface" / "hub")
    host: str = "0.0.0.0"
    port: int = 18002
    n_gpu_layers: int = -1
    n_ctx: int = 4096

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LLM_")


settings = Settings()


def model_path() -> str:
    """Resolve absolute path to downloaded GGUF model file."""
    return str(Path(settings.model_cache_dir) / settings.gguf_filename)
