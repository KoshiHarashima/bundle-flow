# BundleFlow Colab Setup Guide

**🎯 唯一の入口ドキュメント - 他分野エンジニアでも迷わず再現・検証**

## 📋 目次

- [A) GPU環境確認](#a-gpu環境確認)
- [B) CUDA版Torchの入れ方](#b-cuda版torchの入れ方)
- [C) パッケージインストール](#c-パッケージインストール)
- [D) 最小の動作確認](#d-最小の動作確認)
- [E) よくある落とし穴](#e-よくある落とし穴)
- [🚀 Stage1/2 クイックスタート](#-stage12-クイックスタート)
- [🔬 再現性・GPUサニティ](#-再現性gpuサニティ)
- [🛠️ トラブルシューティング](#️-トラブルシューティング)

---

## A) GPU環境確認

### Colab環境でのGPU確認

1. **Runtime設定**
   ```
   Runtime → Change runtime type → GPU（A100/T4）
   ```

2. **GPU確認**
   ```bash
   !nvidia-smi
   ```
   
   **期待出力例:**
   ```
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  Tesla A100-SXM4-40GB  On  | 00000000:00:04.0 Off |                    0 |
   | N/A   45C    P0    42W / 400W |      0MiB / 40960MiB |      0%      Default |
   |                               |                      |             Disabled |
   +-------------------------------+----------------------+----------------------+
   ```

---

## B) CUDA版Torchの入れ方

### 固定手順（失敗しない）

```bash
# 1. CUDA版PyTorchを明示的にインストール
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "torchvision" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "torchaudio" --index-url https://download.pytorch.org/whl/cu121

# 2. 環境確認
python tools/envcheck.py
```

**期待出力:**
```
Python: 3.10.12 Torch: 2.5.1
CUDA available: True
GPU: Tesla A100-SXM4-40GB
Matmul OK on cuda
```

---

## C) パッケージインストール

### パッケージ化＋エントリポイント

```bash
# BundleFlowをインストール（相対パスimportの崩れを根絶）
python -m pip install -e .
```

**これで以下が利用可能:**
- `bundleflow-stage1` - Stage1学習（Flow初期化）
- `bundleflow-stage2` - Stage2学習（Menu最適化）

**新しいAPI構造:**
- `bundleflow.models.BundleFlow` - 速度場ネットワーク
- `bundleflow.models.MenuElement` - メニュー要素
- `bundleflow.models.Mechanism` - 全メニュー
- `bundleflow.valuation.XORValuation` - 評価関数

---

## D) 最小の動作確認

### スモークテスト

```bash
# 1. 基本動作確認
python - <<'PY'
from bundleflow.valuation.valuation import XORValuation
import torch
s = torch.zeros(6); s[0] = 1
v = XORValuation.from_bundle_list(6, [([1], 1.0)])
print("✅ OK:", isinstance(v.value(s), float))
PY

# 2. 新しいAPI構造のテスト
python - <<'PY'
from bundleflow.models.flow import BundleFlow
from bundleflow.models.menu import MenuElement, Mechanism
from bundleflow.valuation.valuation import XORValuation
import torch

# 速度場のテスト
flow = BundleFlow(m=5)
x = torch.randn(3, 5)
t = torch.rand(3)
v = flow.velocity(x, t)
print("✅ BundleFlow velocity shape:", v.shape)

# メニュー要素のテスト
elem = MenuElement(m=5, D=3)
price = elem.price()
print("✅ MenuElement price:", price.item())

# 評価関数のテスト
val = XORValuation.from_bundle_list(5, [([1, 2], 1.0)])
bundle = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
value = val.value(bundle)
print("✅ XORValuation value:", value)

print("🎉 新しいAPI構造が正常に動作しています！")
PY

# 3. 設定ファイル確認
cat conf/stage1.yaml
cat conf/stage2.yaml
```

---

## E) よくある落とし穴

### 1. ノートブック／端末のPython実体が異なる

**問題:** `torch.cuda.is_available()==False` になる

**解決:**
```bash
# ❌ 間違い
pip install torch

# ✅ 正しい（必ず python -m pip を使用）
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
```

**確認方法:**
```bash
which python
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 相対パスimportの崩れ

**問題:** `ModuleNotFoundError: No module named 'bf'`

**解決:** 必ず `pip install -e .` を実行

### 3. 決定論的設定の不備

**問題:** 再現性が保証されない

**解決:** 環境チェックツールで確認
```bash
python tools/envcheck.py
```

---

## 🚀 Stage1/2 クイックスタート

### Stage1学習（Flow初期化）

```bash
# 設定ファイルを使用（推奨）
bundleflow-stage1 --cfg conf/stage1.yaml

# 期待ログ:
# [Stage1] device=cuda torch=2.5.1
# [100/60000] loss=1.2345
# ...
# Saved final: checkpoints/flow_stage1_final.pt
```

### Stage2学習（Menu最適化）

```bash
# 設定ファイルを使用（推奨）
bundleflow-stage2 --cfg conf/stage2.yaml

# 期待ログ:
# [Stage2] device=cuda torch=2.5.1
# [100/20000] rev=0.1234, rev@hard=0.0987, match_rate=0.0234
# ...
# Saved final: checkpoints/menu_stage2_final.pt
```

### 小規模テスト（CPU可）

```bash
# Stage1小規模
python - <<'PY'
from omegaconf import OmegaConf as O
c = O.load('conf/stage1.yaml')
c.iters = 2000; c.batch = 256
O.save(c, 'conf/stage1_quick.yaml')
PY
bundleflow-stage1 --cfg conf/stage1_quick.yaml

# Stage2小規模
python - <<'PY'
from omegaconf import OmegaConf as O
c = O.load('conf/stage2.yaml')
c.K = 128; c.iters = 2000; c.batch = 64
O.save(c, 'conf/stage2_quick.yaml')
PY
bundleflow-stage2 --cfg conf/stage2_quick.yaml
```

---

## 🔬 再現性・GPUサニティ

### 決定論的設定（デフォルトON）

```python
import os, torch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

### 乱数固定

```python
from bundleflow.train.stage1 import seed_all
seed_all(42)
```

### 環境チェックツール

```bash
python tools/envcheck.py
```

**出力例:**
```
Python: 3.10.12 Torch: 2.5.1
CUDA available: True
GPU: Tesla A100-SXM4-40GB
MPS available: False
Matmul OK on cuda
```

---

## 🛠️ トラブルシューティング

### GPU未使用の対策

1. **CUDA wheel再インストール**
   ```bash
   python -m pip uninstall torch torchvision torchaudio -y
   python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Python実体確認**
   ```bash
   which python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **環境チェック**
   ```bash
   python tools/envcheck.py > envcheck.log 2>&1
   cat envcheck.log
   ```

### Stage2の数値安定

1. **β≥0制約**
   - `softplus`関数で実装済み
   - 暴走抑止効果

2. **指数重みの安定化**
   - `log-sum-exp`で実装済み
   - オーバーフロー防止

3. **ウォームアップ**
   - 最初のN千iterはβ固定
   - λ（SoftMax）を1e-3→0.2へ

4. **マッチ率監視**
   - 連続0%ならμ再初期化
   - 合成XORの原子サイズを小さめ（期待5-8）に設定

### 学習信号の確保

1. **合成XORの原子サイズ**
   - 小さめ（期待5-8）に変更済み
   - ランダム半分原子の一致確率向上

2. **マッチ率目標**
   - >1%を目安
   - 統計設定で詰むのを回避

---

## 📊 監視すべきメトリクス

### Stage1（Flow初期化）
- **Coverage probe**: 被覆率が徐々に上昇（最終的に >0.1）
- **Loss**: RectifiedFlow損失が収束
- **Bundle diversity**: 生成されるバンドルの多様性

### Stage2（Menu最適化）
- **match_rate**: (atom_mask & ~bundle_mask)==0 の割合（>1% を目標）
- **rev, rev@hard**: SoftMax 収益と argmax 収益
- **β 統計**: min/median/max（ウォームアップ後に上がり過ぎないか）
- **z 分布**: 要素利用率の偏り（死に要素の検知）

---

## 📝 設定ファイルのカスタマイズ

### A100用最適化設定

```yaml
# conf/stage2_colab_a100.yaml
m: 50
K: 256          # A100用に調整
D: 8
iters: 15000    # 短縮
batch: 128      # A100用に調整
lr: 3e-1
ode_steps: 25
lam_start: 1e-3
lam_end: 2e-1
flow_ckpt: "checkpoints/flow_stage1_final.pt"
seed: 123
device: "auto"
a: 20
n_val: 5000
warmup_iters: 500
match_rate_threshold: 0.01
reinit_on_failure: 100
freeze_beta_iters: 2000  # A100用に延長
auto_optimize: true
use_gumbel: false
warmstart: true
warmstart_grid: 200
reinit_every: 2000
reinit_threshold: 0.01
grad_clip: 0.5           # A100用に調整
ckpt_every: 5000
log_every: 200
eval_n: 1000
atom_size_mode: "small"
cpu: false
```

### 小規模テスト用設定

```yaml
# conf/stage2_small_test.yaml
m: 20           # 小規模
K: 64           # 小規模
D: 4
iters: 1000     # 短縮
batch: 32
lr: 3e-1
ode_steps: 25
lam_start: 1e-3
lam_end: 2e-1
flow_ckpt: "checkpoints/flow_stage1_final.pt"
seed: 123
device: "auto"
a: 10           # 小規模
n_val: 1000     # 小規模
warmup_iters: 100
match_rate_threshold: 0.01
reinit_on_failure: 50
freeze_beta_iters: 200
auto_optimize: true
use_gumbel: false
warmstart: true
warmstart_grid: 50
reinit_every: 500
reinit_threshold: 0.01
grad_clip: 1.0
ckpt_every: 500
log_every: 50
eval_n: 100
atom_size_mode: "small"
cpu: false
```

---

## 🎯 完全な実行例（Colab A100）

```bash
# 1. 環境セットアップ
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python tools/envcheck.py
python -m pip install -e .

# 2. Stage1実行（Flow初期化）
bundleflow-stage1 --cfg conf/stage1.yaml

# 3. Stage2実行（Menu最適化）
bundleflow-stage2 --cfg conf/stage2.yaml

# 4. 結果確認
ls -la checkpoints/
```

---

## 📚 参考文献・技術ノート

- [MODEL.md](MODEL.md) - モデル記号と目的のドキュメント
- [Rectified Flow for Economists](RECTIFIED_FLOW_FOR_ECONOMISTS.md)
- [Gumbel-Softmax Solution](GUMBEL_SOFTMAX_SOLUTION.md)
- [Technical Issues Analysis](TECHNICAL_ISSUES_ANALYSIS.md)
- [Implementation Report](IMPLEMENTATION_REPORT.md)

---

## 🚀 新しいAPI構造の使用例

### 基本的な使用方法

```python
from bundleflow.models.flow import BundleFlow
from bundleflow.models.menu import MenuElement, Mechanism
from bundleflow.valuation.valuation import XORValuation
import torch

# 1. 速度場の初期化
m = 10  # 商品数
flow = BundleFlow(m=m)

# 2. メニュー要素の作成
K = 5   # メニュー要素数
D = 8   # 初期分布の混合成分数
menu = [MenuElement(m=m, D=D) for _ in range(K)]

# 3. 評価関数の作成
atoms = [([1, 2, 3], 5.0), ([4, 5], 3.0), ([6, 7, 8, 9], 8.0)]
valuation = XORValuation.from_bundle_list(m, atoms)

# 4. メカニズムの作成
mechanism = Mechanism(flow, menu)

# 5. 期待収入の計算
revenue = mechanism.expected_revenue([valuation])
print(f"期待収入: {revenue.item():.4f}")

# 6. ハード割当での結果
result = mechanism.argmax_menu([valuation])
print(f"選択されたメニュー: {result['assignments'].item()}")
print(f"効用: {result['utilities'].item():.4f}")
print(f"価格: {result['prices'].item():.4f}")
print(f"収入: {result['revenue'].item():.4f}")
```

### 後方互換性

既存のコードも引き続き動作します：

```python
# 旧API（後方互換性あり）
from bundleflow import FlowModel, MenuElement, XORValuation

# 新API（推奨）
from bundleflow.models import BundleFlow, MenuElement, Mechanism
from bundleflow.valuation import XORValuation
```

---

**🎉 これで他分野のエンジニアでも迷わず再現・検証できます！**