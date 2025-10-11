# Dev Container Setup for BundleFlow

このDev ContainerはGPU（NVIDIA CUDA）をサポートしています。

## 前提条件

ホストマシンに以下がインストールされている必要があります：

1. **Docker Desktop**（macOSの場合）または **Docker Engine**（Linuxの場合）
2. **NVIDIA Docker runtime**（Linux/WSL2の場合）
3. **NVIDIA Driver**（最新版推奨）

### macOSでの注意

⚠️ **重要**: macOSではNVIDIA GPUのパススルーはサポートされていません。
macOSで開発している場合は、以下のいずれかを検討してください：

- リモートGPUサーバーに接続してDev Containerを使用
- AWS/GCP/Azure等のクラウドGPUインスタンスを使用
- Linux/WSL2環境に移行

### Linuxでのセットアップ

```bash
# NVIDIA Container Toolkitのインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 使い方

1. Cursor/VS Codeでプロジェクトを開く
2. コマンドパレット（Cmd/Ctrl + Shift + P）を開く
3. "Dev Containers: Reopen in Container" を選択
4. コンテナが起動するまで待つ

## GPU確認

コンテナ内で以下のコマンドを実行してGPUが認識されているか確認：

```bash
# NVIDIAドライバーの確認
nvidia-smi

# PyTorchからのGPU確認
check-gpu

# または手動で
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## トラブルシューティング

### "could not select device driver" エラー

NVIDIA Container Toolkitがインストールされていない可能性があります。
上記のLinuxセットアップ手順を実行してください。

### GPUが認識されない

1. ホストでGPUドライバーが正しくインストールされているか確認：
   ```bash
   nvidia-smi
   ```

2. Dockerでnvidia-runtimeが使えるか確認：
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. それでも解決しない場合は、Dockerを再起動してみてください。

