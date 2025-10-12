#!/usr/bin/env python3
"""
BundleFlow Environment Check Tool
CUDA/デバイス/バージョン/決定論フラグとサンプル matmul を出力
Colab/Terminal差異を潰すための包括的な環境チェック
"""

import os
import platform
import torch
import subprocess as sp

def main():
    print("Python:", platform.python_version())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("MPS available:", torch.backends.mps.is_available())
    print("CUBLAS_WORKSPACE_CONFIG:", os.getenv("CUBLAS_WORKSPACE_CONFIG"))
    print("Deterministic:", torch.are_deterministic_algorithms_enabled())
    
    # 1ステップ matmul 実行で実動確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8192, 8192, device=device)
    y = x @ x.t()
    print("matmul OK, device:", y.device)

if __name__ == "__main__":
    main()