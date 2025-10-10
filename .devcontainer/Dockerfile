# 例：CUDA 12.x + PyTorch 2.x のタグを選ぶ
FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /bundle-flow
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git build-essential curl vim && rm -rf /var/lib/apt/lists/*

# 最小限のOSパッケージ（必要時のみ）
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git build-essential curl vim \
 && rm -rf /var/lib/apt/lists/*

# 追加Pythonツール
RUN pip install -U pip uv

# Python依存をビルド時に固定
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# 確認（ログにCUDA可否を出す）
RUN python - <<'PY'\nimport torch;print('cuda:',torch.cuda.is_available())\nPY
