.PHONY: all asr llm tts livekit agent stop setup download

all:
	bash scripts/start_all.sh

asr:
	bash scripts/start_asr.sh

llm:
	bash scripts/start_llm.sh

tts:
	bash scripts/start_tts.sh

livekit:
	bash scripts/start_livekit.sh

agent:
	bash scripts/start_agent.sh

web:
	bash scripts/start_web.sh

stop:
	bash scripts/stop_all.sh

setup:
	bash scripts/setup.sh

download:
	bash scripts/download_models.sh

test:
	source .venv/bin/activate && pytest tests/ -v
