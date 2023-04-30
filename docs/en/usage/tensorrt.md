# TensorRT

TensorRT is a high-speed inference engine provided by NVIDIA.

## Requirements

- `TensorRT` == 8.6.0
- CUDNN == 8.8.0
- CUDA >= 11.0

## Usage

### Windows

1. Rewrite `webui-user.bat` as follows

```bat
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--tensorrt --reinstall-torch
set TORCH_COMMAND=pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

call launch.bat
```

2. Run `launch-user.bat`

### Linux or MacOS

1. Rewrite `webui-user.sh` as follows

```sh
# export COMMANDLINE_ARGS="--tensorrt"
```

↓

```sh
export COMMANDLINE_ARGS="--tensorrt --reinstall-torch"
export TORCH_COMMAND="pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
```

2. Run `launch-user.sh`
