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
    print("MPS available:", torch.backends.mps.is_available())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    x = torch.randn(2048,2048, device="cuda" if torch.cuda.is_available() else "cpu")
    y = x @ x.t()
    print("Matmul OK on", y.device)

if __name__ == "__main__":
    main()