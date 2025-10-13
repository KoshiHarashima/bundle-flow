# PyTorch + CUDA環境
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /bundle-flow

# CUDAが利用可能かチェックするためのスクリプト
RUN echo '#!/bin/bash\n\
    echo "=== GPU Status ==="\n\
    nvidia-smi\n\
    echo ""\n\
    echo "=== PyTorch GPU Status ==="\n\
    python -c "import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\"); print(f\"cuDNN version: {torch.backends.cudnn.version()}\"); print(f\"Number of GPUs: {torch.cuda.device_count()}\"); [print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())]"\n\
    ' > /usr/local/bin/check-gpu && chmod +x /usr/local/bin/check-gpu

# デフォルトのシェルをbashに設定
SHELL ["/bin/bash", "-c"]

# Pythonのパスを設定
ENV PATH="/opt/conda/bin:${PATH}"
