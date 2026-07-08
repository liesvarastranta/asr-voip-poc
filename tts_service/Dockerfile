FROM --platform=linux/arm64 nvidia/cuda:12.8-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY ref_audio/ ref_audio/

EXPOSE 18003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "18003"]
