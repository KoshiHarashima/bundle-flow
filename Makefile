# BundleFlow Makefile
# 5分で再現・検証できる最小セット

.PHONY: help env test format lint clean reproduce

help:
	@echo "BundleFlow - Available commands:"
	@echo "  make env        - 環境構築"
	@echo "  make test       - テスト実行"
	@echo "  make format     - コードフォーマット"
	@echo "  make lint       - リントチェック"
	@echo "  make clean      - クリーンアップ"
	@echo "  make reproduce  - 5分で再現（小規模）"

env:
	pip install -e .
	pip install pytest ruff black

test:
	pytest -q

format:
	black .
	ruff --fix .

lint:
	ruff check .
	black --check .

clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

reproduce:
	@echo "🚀 5分で再現（小規模設定）"
	@echo "=========================="
	@echo "1. Stage1小規模実行..."
	python - <<'PY'
from omegaconf import OmegaConf as O
c = O.load('conf/stage1.yaml')
c.iters = 100; c.batch = 64
O.save(c, 'conf/stage1_reproduce.yaml')
PY
	bundleflow-stage1 --cfg conf/stage1_reproduce.yaml
	@echo "2. Stage2小規模実行..."
	python - <<'PY'
from omegaconf import OmegaConf as O
c = O.load('conf/stage2.yaml')
c.K = 32; c.iters = 100; c.batch = 32
O.save(c, 'conf/stage2_reproduce.yaml')
PY
	bundleflow-stage2 --cfg conf/stage2_reproduce.yaml
	@echo "3. 結果確認..."
	ls -la checkpoints/
	@echo "✅ 再現完了！"