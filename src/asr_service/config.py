from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_id: str = "Qwen/Qwen3-ASR-1.7B-hf"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    chunk_ms: int = 500
    sample_rate: int = 16000
    max_session_s: int = 300
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ASR_")

settings = Settings()
