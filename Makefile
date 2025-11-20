# Makefile untuk Windows PowerShell
# Run: make <target>

.PHONY: help install init scrape preprocess train predict stats docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make init         - Initialize project"
	@echo "  make scrape       - Scrape reviews"
	@echo "  make preprocess   - Preprocess data"
	@echo "  make train        - Train model"
	@echo "  make predict      - Run predictions"
	@echo "  make stats        - Show statistics"
	@echo "  make docker-up    - Start Docker containers"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make dvc-repro    - Run DVC pipeline"
	@echo "  make clean        - Clean temporary files"

install:
	pip install -r requirements.txt
	pip install -r requirements-cli.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

init:
	python cli.py init

scrape:
	python cli.py scrape

preprocess:
	python cli.py preprocess

train:
	python cli.py train

predict:
	python cli.py predict

stats:
	python cli.py stats

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

dvc-repro:
	dvc repro

dvc-status:
	dvc status

dvc-metrics:
	dvc metrics show

clean:
	python -c "import shutil; import os; [shutil.rmtree(d, ignore_errors=True) for d in ['__pycache__', '.pytest_cache', '.mypy_cache', 'dist', 'build']]; [os.remove(f) for f in ['.coverage'] if os.path.exists(f)]"
	Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
	Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force

test:
	python -m pytest tests/ -v

run-scheduler:
	python src/scheduler/main.py

jupyter:
	jupyter notebook notebooks/
