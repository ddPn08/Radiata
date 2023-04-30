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
set COMMANDLINE_ARGS=--tensorrt

call launch.bat
```

2. Run `launch-user.bat`

### Linux or MacOS

1. Rewrite `webui-user.sh` as follows

```sh
# export COMMANDLINE_ARGS=""
```

↓

```sh
export COMMANDLINE_ARGS="--tensorrt"
```

2. Run `launch-user.sh`
