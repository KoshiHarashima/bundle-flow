# BundleFlow Colab Setup Guide

## 🚀 A100用の固定手順（Colab環境）

### Step 1: CUDA PyTorch の明示的インストール

```bash
# A100用: CUDA 12.1 wheel を明示
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "torchvision" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "torchaudio" --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: 環境確認と設定

```python
# 環境確認
python - <<'PY'
import torch, os
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 決定論的設定
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
print("✅ Deterministic algorithms enabled")
PY
```

### Step 3: 環境チェックツールの実行

```bash
# 包括的な環境チェック
python tools/envcheck.py
```

### Step 4: Stage2学習の実行

```bash
# A100最適化版
python -m src.train_stage2 \
    --flow_ckpt checkpoints/flow_stage1_final.pt \
    --m 50 --K 256 --D 8 \
    --iters 15000 --batch 128 \
    --ode_steps 25 --atom_size_mode small \
    --warmstart --reinit_every 1000 \
    --log_every 20 \
    --out_dir checkpoints_stage2_colab_a100 \
    --auto_optimize \
    --a 20 \
    --freeze_beta_iters 2000 \
    --grad_clip 0.5
```

## 🔧 トラブルシューティング

### GPU未使用の対策（チェックリスト）

1. **nvidia-smi と PyTorch の両方確認**
   ```bash
   !nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **CUDA wheel を明示して再インストール**
   ```bash
   python -m pip uninstall torch torchvision torchaudio -y
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **端末とノートで同じ Python を使用**
   ```bash
   which python
   python -m pip install  # pip 単体禁止
   ```

4. **モデルとテンソルの device 混在を禁止**
   ```python
   # 正しい作法
   model = model.to(device)
   tensor = tensor.to(device)
   ```

5. **環境チェックツールの出力を必ずログに残す**
   ```bash
   python tools/envcheck.py > envcheck.log 2>&1
   ```

## 📊 監視すべきメトリクス（Stage2）

- **match_rate**: (atom_mask & ~bundle_mask)==0 の割合（>1% を目標）
- **rev, rev@hard**: SoftMax 収益と argmax 収益
- **β 統計**: min/median/max（ウォームアップ後に上がり過ぎないか）
- **z 分布**: 要素利用率の偏り（死に要素の検知）
- **キャッシュヒット**: Tr(Q(μ_d)) と ∫η の前計算再利用率

## 🧪 再現性（最小セット）

1. **乱数固定 + 決定論**
   ```python
   torch.manual_seed(42)
   torch.use_deterministic_algorithms(True)
   os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
   ```

2. **依存の固定**
   ```bash
   pip freeze > requirements.txt
   ```

3. **実験ログ**
   - ハイパラ、メトリクス、チェックポイント、Git SHA を保存
