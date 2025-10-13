#!/usr/bin/env python3
"""
BundleFlow Environment Check Tool
CUDA/デバイス/バージョン/決定論フラグとサンプル matmul を出力
Colab/Terminal差異を潰すための包括的な環境チェック
"""

import os, platform, torch

def main():
    print("Python:", platform.python_version(), "Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    print("MPS available:", torch.backends.mps.is_available())
    
    # 決定論的設定（デフォルトON）
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    # 簡単なmatmulテスト
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2048, 2048, device=device)
    y = x @ x.t()
    print("Matmul OK on", y.device)
    
    # 再現性テスト
    torch.manual_seed(42)
    a = torch.randn(10, 10, device=device)
    torch.manual_seed(42)
    b = torch.randn(10, 10, device=device)
    print("Reproducibility:", torch.allclose(a, b))

if __name__ == "__main__":
    main()