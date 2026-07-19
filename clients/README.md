# ASR Service — Client Example

## Requirements

```bash
pip install httpx soundfile numpy
```

## Usage

```bash
python python_client.py sample.wav
```

Output: SSE events as they arrive — `partial` for ongoing transcription, `final` for completed utterance.
