FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG work_dir="/workspace"

# ENV PYTHON_VERSION 3.10.9
# ENV HOME /root
# ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
# ENV PATH $PYTHON_ROOT/bin:$PATH
# ENV PYENV_ROOT $HOME/.pyenv

#gitのインストール
RUN apt-get update -y && apt-get install -y build-essential vim \
    wget curl git zip gcc make openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    libfreetype6-dev libffi-dev liblzma-dev

# Install nodejs for JupyterLab extension
RUN curl -sL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# ワークディレクトリを設定
WORKDIR /bundle-flow

# uvをインストール
RUN pip install uv
