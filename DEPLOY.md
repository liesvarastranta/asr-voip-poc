# Deploy to ASUS Ascent GX10

Target: `10.9.23.200` (GX10 — NVIDIA GB10 Grace Blackwell, ARM64, CUDA)

## Prerequisites (GX10)

```bash
# NVIDIA Container Toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify CUDA in Docker
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu24.04 nvidia-smi
```

## Deploy

```bash
# 1. Clone
ssh amma@10.9.23.200
git clone git@github.com:liesvarastranta/asr-voip-poc.git
cd asr-voip-poc

# 2. Download model (~4 GB) to persistent volume
docker compose run --rm asr python scripts/download_model.py

# 3. Start service
docker compose up --build -d

# 4. Verify
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model":"Qwen/Qwen3-ASR-1.7B-hf","device":"cuda:0"}

curl -F file=@test.wav http://localhost:8000/v1/asr/transcribe
# {"text":"hasil transkripsi...","language":"id","duration_ms":5000,"processing_ms":420}

# 5. Real-time streaming test
pip install httpx soundfile numpy
python clients/python_client.py test.wav

# 6. Benchmark
python scripts/benchmark.py
# (set BENCH_FILES=audio1.wav,audio2.wav)
```

## Remote access

```bash
# From other machines:
curl http://10.9.23.200:8000/health
curl http://10.9.23.200:8000/docs
```

## Troubleshoot

```bash
# Check logs
docker compose logs -f asr

# Check GPU
docker compose exec asr nvidia-smi

# Rebuild after code changes
git pull && docker compose up --build -d
```
