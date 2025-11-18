.PHONY: install test clean train lint format help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install -r requirements.txt
	pip install -e .

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term

train-quick:  ## Quick training run (reduced epochs)
	python scripts/train.py --data data/synthetic_data.h5 --quick

train:  ## Full training run
	python scripts/train.py --data data/synthetic_data.h5

clean:  ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -f .coverage

lint:  ## Run linting
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

format:  ## Format code
	black src/ tests/ --line-length=100
	isort src/ tests/

notebook:  ## Launch Jupyter notebook
	jupyter notebook notebooks/

create-demo-data:  ## Create synthetic demo dataset
	python -c "import h5py, numpy as np; \
	n=10000; \
	h5py.File('data/synthetic_data.h5', 'w').create_dataset('windows', data=np.random.randn(n, 820, 3).astype('f4')); \
	h5py.File('data/synthetic_data.h5', 'a').create_dataset('labels', data=np.random.choice([0,1,2,3], n, p=[0.3,0.4,0.2,0.1]).astype('i4'))"
	@echo "Created data/synthetic_data.h5"
