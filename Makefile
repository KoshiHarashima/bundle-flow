.PHONY: env stage1 stage2 test fmt lint

env:
	python -m pip install -e .
	python -m pip install ruff black pytest omegaconf mlflow

stage1:
	bundleflow-stage1 --cfg conf/stage1.yaml

stage2:
	bundleflow-stage2 --cfg conf/stage2.yaml

test:
	pytest -q

fmt:
	black .

lint:
	ruff check .
