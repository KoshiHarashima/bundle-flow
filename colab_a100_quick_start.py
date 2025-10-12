#!/usr/bin/env python3
"""
Colab A100用のクイックスタートスクリプト
CUDA版PyTorchのインストールから実行まで自動化
"""

import subprocess
import sys
import torch

def check_and_install_cuda_pytorch():
    """CUDA版PyTorchの確認とインストール"""
    print("🔍 Checking PyTorch installation...")
    
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("❌ CUDA not available! Installing CUDA PyTorch...")
            return False
            
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False

def install_cuda_pytorch():
    """CUDA版PyTorchをインストール"""
    print("🚀 Installing CUDA PyTorch...")
    
    try:
        # 既存のPyTorchをアンインストール
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      check=True)
        
        # CUDA版をインストール
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                       "--index-url", "https://download.pytorch.org/whl/cu121"], 
                      check=True)
        
        print("✅ CUDA PyTorch installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install CUDA PyTorch: {e}")
        return False

def run_stage2_training():
    """Stage2学習を実行"""
    print("🎯 Starting Stage2 training with Colab A100 optimizations...")
    
    cmd = [
        sys.executable, "-m", "src.train_stage2",
        "--flow_ckpt", "checkpoints/flow_stage1_final.pt",
        "--m", "50",
        "--K", "128", 
        "--D", "4",
        "--iters", "10000",
        "--batch", "64",
        "--ode_steps", "25",
        "--atom_size_mode", "small",
        "--warmstart",
        "--reinit_every", "1000",
        "--log_every", "20",
        "--out_dir", "checkpoints_stage2_colab_a100",
        "--auto_optimize"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    print("🚀 Colab A100 Quick Start Script")
    print("=" * 50)
    
    # 1. PyTorchの確認
    if not check_and_install_cuda_pytorch():
        # 2. CUDA版PyTorchのインストール
        if not install_cuda_pytorch():
            print("❌ Failed to install CUDA PyTorch. Exiting.")
            return
        
        # 3. 再確認
        print("🔄 Restarting Python to reload PyTorch...")
        if not check_and_install_cuda_pytorch():
            print("❌ CUDA PyTorch still not available. Exiting.")
            return
    
    # 4. Stage2学習の実行
    run_stage2_training()

if __name__ == "__main__":
    main()
