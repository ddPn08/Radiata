# Deepfloyd IF

IF は、Stability AI 社の Deepfloyd チームが開発した新しい画像生成 AI 技術です。

## 使い方

### Windows

1. `webui-user.bat`を以下のように書き換えます

```bat
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--deepfloyd_if

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
export COMMANDLINE_ARGS="--deepfloyd_if"
```

2. `launch-user.sh`を実行します
