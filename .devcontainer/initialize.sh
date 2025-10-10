#!/bin/bash

echo "あなたの名前を入力してください（コンテナ名に使用されます）："
read username

# 入力が空の場合はエラーメッセージを表示して終了
if [ -z "$username" ]; then
  echo "エラー: 名前が入力されませんでした。プロセスを終了します。"
  exit 1
else
  echo "ようこそ、$username さん！"
fi

# .envファイルを作成または上書き
echo "USER_NAME=\"$username\"" > .devcontainer/.env

# Gitの設定を取得
echo "Gitのユーザー名を入力してください："
read git_username

# 入力が空の場合はエラーメッセージを表示
if [ -z "$git_username" ]; then
  echo "警告: Gitのユーザー名が入力されませんでした。Gitの設定はスキップされます。"
else
  echo "Gitのメールアドレスを入力してください："
  read git_email

  if [ -z "$git_email" ]; then
    echo "警告: Gitのメールアドレスが入力されませんでした。Gitの設定はスキップされます。"
  else
    # Gitの設定情報を.envファイルに保存（コンテナ内で利用するため）
    echo "GIT_USER_NAME=\"$git_username\"" >> .devcontainer/.env
    echo "GIT_USER_EMAIL=\"$git_email\"" >> .devcontainer/.env
    echo "Gitの設定情報を保存しました。コンテナ起動後に適用されます。"
  fi
fi

# NVIDIAドライバとCUDAの確認
if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIAドライバが見つかりました。情報を表示します:"
  nvidia-smi
else
  echo "警告: NVIDIAドライバが見つかりません。GPUが利用できない可能性があります。"
fi

if command -v nvcc &> /dev/null; then
  echo "CUDAが見つかりました。バージョン情報:"
  nvcc --version
else
  echo "警告: CUDAが見つかりません。"
fi

# 実行権限を確保
chmod +x .devcontainer/initialize.sh

echo "設定が完了しました。devcontainerを起動します..."