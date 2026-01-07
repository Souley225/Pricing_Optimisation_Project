.PHONY: install lint format typecheck test clean data features train evaluate optimize api ui all

# Installation
install:
	poetry install

# Quality
lint:
	poetry run ruff check .

format:
	poetry run ruff format .

typecheck:
	poetry run mypy src/

# Testing
test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term

# Pipeline steps
data:
	poetry run python -m src.data.make_dataset

features:
	poetry run python -m src.features.build_features

train:
	poetry run python -m src.models.train

evaluate:
	poetry run python -m src.models.evaluate

optimize:
	poetry run python -m src.models.optimize_prices

# Services
api:
	poetry run uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

ui:
	poetry run streamlit run src/ui/app.py --server.port 8501

mlflow-server:
	poetry run mlflow server --host 0.0.0.0 --port 5000

# DVC
dvc-repro:
	poetry run dvc repro

dvc-push:
	poetry run dvc push

dvc-pull:
	poetry run dvc pull

# Docker
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true

# Full pipeline
all: lint typecheck test data features train evaluate optimize
