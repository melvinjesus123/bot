# Comandos comunes
.PHONY: install lint test demo ingest-once ingest-forever backtest docker-build docker-run

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src tests --max-line-length=120

test:
	pytest -q

demo:
	python scripts/run_demo.py

ingest-once:
	python src/ingest/ccxt_ingest.py

ingest-forever:
	python -c "import asyncio; from src.ingest.ccxt_ingest import run_forever; asyncio.run(run_forever())"

backtest:
	python scripts/run_backtest.py

docker-build:
	docker build -f docker/Dockerfile -t crypto-bot:latest .

docker-run:
	docker compose -f docker/docker-compose.yml up --build
