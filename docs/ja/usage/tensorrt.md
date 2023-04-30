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
set COMMANDLINE_ARGS=--tensorrt

call launch.bat
```

2. `launch-user.bat`を実行します

### Linux or MacOS

1. `webui-user.sh`を以下のように書き換えます

```sh
# export COMMANDLINE_ARGS=""
```

↓

```sh
export COMMANDLINE_ARGS="--tensorrt"
```

2. `launch-user.sh`を実行します
