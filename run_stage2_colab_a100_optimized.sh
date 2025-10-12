#!/bin/bash
# Colab A100用の最適化されたStage2実行スクリプト

echo "🚀 Colab A100 Optimized Stage2 Training Script"
echo "=============================================="

# CUDA版PyTorchのインストール確認
echo "🔍 Checking PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available! Installing CUDA PyTorch...')
    import subprocess
    subprocess.run(['pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'])
    subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121'])
    print('✅ CUDA PyTorch installed! Please restart the script.')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "🔄 Restarting after PyTorch installation..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
fi

echo ""
echo "🎯 Starting Stage2 training with Colab A100 optimizations..."

# Colab A100用の最適化パラメータで実行
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
    --out_dir checkpoints_stage2_colab_a100 \
    --auto_optimize

echo "✅ Training completed!"
