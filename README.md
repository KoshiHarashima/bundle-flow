# BundleFlow

Rectified Flow–based menus for combinatorial auctions

## 🚀 5分で動作確認

### 最小手順（端末/Colab共通）

```bash
git clone https://github.com/KoshiHarashima/bundle-flow
cd bundle-flow
python -m pip install -e .
python tools/envcheck.py
```

### Colab環境での実行

```bash
# 1) Torch CUDA wheel を明示（環境に合わせて cu118/cu121 等）
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# 2) 再現性フラグ（任意）
python - <<'PY'
import os, torch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
print("deterministic set")
PY

# 3) サニティ
python tools/envcheck.py
```

### Stage1 (小規模)

```bash
python - <<'PY'
from omegaconf import OmegaConf as O
c=O.load('conf/stage1.yaml')
c.iters=2000
c.batch=256
O.save(c, 'conf/stage1_quick.yaml')
PY
bundleflow-stage1 --cfg conf/stage1_quick.yaml
```

### Stage2 (小規模)

```bash
python - <<'PY'
from omegaconf import OmegaConf as O
c=O.load('conf/stage2.yaml')
c.K=128
c.iters=2000
c.batch=64
O.save(c, 'conf/stage2_quick.yaml')
PY
bundleflow-stage2 --cfg conf/stage2_quick.yaml
```

## 📊 想定ログ断片（確認ポイント）

```
[Stage1] device=cuda torch=2.5.1  GPU=A100-SXM4-40GB
[200/2000] loss=1.72 ...
...
Saved final: checkpoints/flow_stage1_final.pt

[Stage2] device=cuda torch=2.5.1
[200/2000] LRev=-0.183 lam=0.012 match_rate=1.6% beta_med=0.42 ...
```

## 🛠️ 開発者向け

### 環境構築

```bash
make env
```

### テスト

```bash
make test
```

### フォーマット

```bash
make fmt
make lint
```

### 実行

```bash
make stage1
make stage2
```

## 📁 プロジェクト構造

```
bundle-flow/
├─ bundleflow/                  # メインパッケージ
│  ├─ __init__.py
│  ├─ flow.py
│  ├─ menu.py
│  ├─ valuation.py
│  ├─ data.py
│  ├─ utils.py
│  └─ cli/
│     ├─ stage1.py
│     └─ stage2.py
├─ bf/                          # 互換レイヤ（当面残す）
│  └─ __init__.py
├─ conf/                        # 設定
│  ├─ stage1.yaml
│  └─ stage2.yaml
├─ tools/
│  └─ envcheck.py               # GPU/環境サニティ
├─ tests/
│  └─ test_smoke.py
├─ .github/workflows/ci.yml
├─ Makefile
├─ pyproject.toml
└─ README.md
```

## 🔧 設定

設定はYAMLファイルで管理されます：

- `conf/stage1.yaml` - Stage1の設定
- `conf/stage2.yaml` - Stage2の設定

設定の差分上書き例：

```python
from omegaconf import OmegaConf as O
c=O.load('conf/stage2.yaml')
c.K=128
O.save(c, 'tmp.yaml')
bundleflow-stage2 --cfg tmp.yaml
```

## 🎯 主要機能

- **Stage1**: Rectified Flow による分布学習
- **Stage2**: メニュー最適化（収益最大化）
- **GPU最適化**: CUDA/MPS/CPU自動検出
- **数値安定性**: log-sum-exp実装
- **再現性**: 決定論的アルゴリズム対応