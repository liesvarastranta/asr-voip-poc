"""Launch llama-cpp-python OpenAI-compatible server."""
import subprocess
import sys
from .config import settings, model_path


def main() -> None:
    path = model_path()
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", path,
        "--n_gpu_layers", str(settings.n_gpu_layers),
        "--n_ctx", str(settings.n_ctx),
        "--host", settings.host,
        "--port", str(settings.port),
    ]
    print(f"Starting llama-cpp-python server: {cmd}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
