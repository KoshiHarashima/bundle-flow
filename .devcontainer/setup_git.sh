#!/bin/bash

# .envファイルから変数を読み込む
if [ -f /workspaces/office-dx-platform/.devcontainer/.env ]; then
  # .envファイルを読み込む（変数にスペースが含まれている場合でも正しく処理できるように）
  set -a
  source /bundle-flow/.devcontainer/.env
  set +a

  # Gitユーザー名とメールアドレスが設定されていれば適用する
  if [ ! -z "$GIT_USER_NAME" ] && [ ! -z "$GIT_USER_EMAIL" ]; then
    echo "コンテナ内のGit設定を適用しています..."
    git config --global user.name "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
    echo "Git設定が完了しました: $GIT_USER_NAME <$GIT_USER_EMAIL>"
  else
    echo "Git設定情報が見つかりません。スキップします。"
    echo "環境変数の内容: GIT_USER_NAME=${GIT_USER_NAME:-未設定}, GIT_USER_EMAIL=${GIT_USER_EMAIL:-未設定}"
  fi
else
  echo ".envファイルが見つかりません。Git設定をスキップします。"
  echo "検索パス: /workspaces/office-dx-platform/.devcontainer/.env"
fi