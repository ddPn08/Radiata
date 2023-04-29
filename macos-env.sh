#!/bin/bash
####################################################################
#                          macOS defaults                          #
# Please modify webui-user.sh to change these instead of this file #
####################################################################

if [[ -x "$(command -v python3.10)" ]]
then
    python_cmd="python3.10"
fi

export COMMANDLINE_ARGS="--skip-torch-cuda-test"
export TORCH_COMMAND="pip install torch torchvision torchaudio"
export PYTORCH_ENABLE_MPS_FALLBACK=1

####################################################################