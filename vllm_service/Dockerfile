FROM --platform=linux/arm64 nvidia/cuda:12.8-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN pip install --no-cache-dir vllm huggingface-hub

COPY download_model.py .

EXPOSE 18002

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen3.6-35B-A3B", \
     "--host", "0.0.0.0", \
     "--port", "18002", \
     "--dtype", "bfloat16", \
     "--gpu-memory-utilization", "0.5"]
