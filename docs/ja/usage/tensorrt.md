# TensorRT

TensorRT は NVIDIA が提供する高速な推論エンジンです。

## Requirements

- `TensorRT` == 8.6.0
- CUDNN == 8.8.0
- CUDA >= 11.0
- pytorch < 2

## 使い方

### Windows

1. `webui-user.bat`を以下のように書き換えます

```bat
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--tensorrt --reinstall-torch
set TORCH_COMMAND=pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
set XFORMERS_COMMAND=pip install xformers==0.0.16

call launch.bat
```

2. `launch-user.bat`を実行します

### Linux or MacOS

1. `webui-user.sh`を以下のように書き換えます

```sh
# export COMMANDLINE_ARGS="--tensorrt"
```

↓

```sh
export COMMANDLINE_ARGS="--tensorrt --reinstall-torch"
export TORCH_COMMAND="pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
export XFORMERS_COMMAND="pip install xformers==0.0.16"
```

2. `launch-user.sh`を実行します
