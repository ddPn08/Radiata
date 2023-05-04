# Deepfloyd IF

IF is a new image generation AI technology developed by the Deepfloyd team at Stability AI.

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
