#!/bin/bash
# スマートなStage2実行スクリプト
# デバイスを自動検出し、最適なパラメータで実行

echo "🚀 Smart Stage2 Training Script"
echo "================================"

# デバイス確認
echo "🔍 Checking PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "🎯 Starting Stage2 training with auto-optimization..."

# 自動最適化を有効にして実行
python3 -m src.train_stage2 \
    --flow_ckpt checkpoints/flow_stage1_final.pt \
    --m 50 \
    --K 128 \
    --D 4 \
    --iters 10000 \
    --batch 64 \
    --ode_steps 25 \
    --atom_size_mode small \
    --warmstart \
    --reinit_every 1000 \
    --log_every 20 \
    --out_dir checkpoints_stage2_smart \
    --auto_optimize

echo "✅ Training completed!"
