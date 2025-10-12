#!/usr/bin/env python3
"""
デバイス移動のテストスクリプト
PyTorchのデバイス移動が正しく動作するかテスト
"""

import torch
import sys

def test_device_availability():
    """デバイスの利用可能性をテスト"""
    print("🔍 Testing device availability...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # デバイス選択
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Selected device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Selected device: {device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Selected device: {device} (no GPU acceleration)")
    
    return device

def test_tensor_movement(device):
    """テンソルのデバイス移動をテスト"""
    print(f"\n🧪 Testing tensor movement to {device}...")
    
    # CPU上でテンソルを作成
    x = torch.randn(100, 100)
    print(f"Original tensor device: {x.device}")
    
    # デバイスに移動
    x_device = x.to(device)
    print(f"Moved tensor device: {x_device.device}")
    
    # 計算テスト
    y = torch.randn(100, 100, device=device)
    z = torch.mm(x_device, y)
    print(f"Matrix multiplication result device: {z.device}")
    print(f"Result shape: {z.shape}")
    
    return True

def test_model_movement(device):
    """モデルのデバイス移動をテスト"""
    print(f"\n🧪 Testing model movement to {device}...")
    
    # 簡単なモデルを作成
    model = torch.nn.Linear(100, 50)
    print(f"Original model device: {next(model.parameters()).device}")
    
    # デバイスに移動
    model = model.to(device)
    print(f"Moved model device: {next(model.parameters()).device}")
    
    # 推論テスト
    x = torch.randn(10, 100, device=device)
    y = model(x)
    print(f"Model output device: {y.device}")
    print(f"Model output shape: {y.shape}")
    
    return True

def test_memory_management(device):
    """メモリ管理をテスト"""
    print(f"\n🧪 Testing memory management on {device}...")
    
    if device.type == "cuda":
        print(f"GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # 大きなテンソルを作成
        x = torch.randn(1000, 1000, device=device)
        print(f"GPU memory after allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # メモリクリーンアップ
        del x
        torch.cuda.empty_cache()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
    elif device.type == "mps":
        print("MPS memory management test...")
        
        # 大きなテンソルを作成
        x = torch.randn(1000, 1000, device=device)
        print(f"MPS tensor created: {x.shape}")
        
        # メモリクリーンアップ
        del x
        torch.mps.empty_cache()
        print("MPS memory cleaned up")
    
    return True

def main():
    print("🚀 PyTorch Device Movement Test")
    print("=" * 50)
    
    try:
        # 1. デバイス利用可能性テスト
        device = test_device_availability()
        
        # 2. テンソル移動テスト
        test_tensor_movement(device)
        
        # 3. モデル移動テスト
        test_model_movement(device)
        
        # 4. メモリ管理テスト
        test_memory_management(device)
        
        print("\n🎉 All tests passed!")
        print(f"✅ Device {device} is ready for training!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
