#!/usr/bin/env python3
"""
デバイス切り替えスクリプト
PyTorchのCPU/GPU版を簡単に切り替える
"""

import subprocess
import sys
import os

def check_current_pytorch():
    """現在のPyTorchバージョンを確認"""
    try:
        import torch
        print(f"🔍 Current PyTorch version: {torch.__version__}")
        print(f"🚀 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_cuda_pytorch():
    """CUDA版PyTorchをインストール"""
    print("🚀 Installing CUDA-enabled PyTorch...")
    try:
        # CPU版をアンインストール
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

def install_cpu_pytorch():
    """CPU版PyTorchをインストール"""
    print("💻 Installing CPU-only PyTorch...")
    try:
        # CUDA版をアンインストール
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      check=True)
        
        # CPU版をインストール
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                       "--index-url", "https://download.pytorch.org/whl/cpu"], 
                      check=True)
        
        print("✅ CPU PyTorch installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install CPU PyTorch: {e}")
        return False

def main():
    print("🔧 PyTorch Device Switcher")
    print("=" * 50)
    
    # 現在の状況を確認
    check_current_pytorch()
    print()
    
    while True:
        print("📋 Options:")
        print("1. Install CUDA PyTorch (for GPU acceleration)")
        print("2. Install CPU PyTorch (for CPU-only)")
        print("3. Check current PyTorch status")
        print("4. Exit")
        
        choice = input("\n🎯 Choose option (1-4): ").strip()
        
        if choice == "1":
            if install_cuda_pytorch():
                print("\n🔄 Restarting Python to reload PyTorch...")
                check_current_pytorch()
            break
            
        elif choice == "2":
            if install_cpu_pytorch():
                print("\n🔄 Restarting Python to reload PyTorch...")
                check_current_pytorch()
            break
            
        elif choice == "3":
            check_current_pytorch()
            print()
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
