import argparse
import os
from typing import *

import toml
from packaging.version import Version

from .version import update

DEFAULT_CONFIG = toml.loads(
    """
version = "0.0.1"
model_dir = "models"
model = "runwayml/stable-diffusion-v1-5"
models = [ "runwayml/stable-diffusion-v1-5" ]

[common]
output-dir-txt2img = "outputs/txt2img"
output-name-txt2img = "{index}-{seed}-{prompt}.png"
output-dir-img2img = "outputs/img2img"
output-name-img2img = "{index}-{seed}-{prompt}.png"
image-browser-dir = "outputs"

[acceleration.tensorrt]
full-acceleration = true
"""
)

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


parser = argparse.ArgumentParser()

parser.add_argument("--config-file", type=str, default="config.toml")

parser.add_argument("--host", type=str, default="")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")

parser.add_argument("--model-dir", type=str, default="models")
parser.add_argument("--hf-token", type=str)

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")

parser.add_argument("--xformers", action="store_true")
parser.add_argument("--tensorrt", action="store_true")
parser.add_argument("--deepfloyd_if", action="store_true")

cmd_opts, _ = parser.parse_known_args(
    os.environ["COMMANDLINE_ARGS"].split(" ")
    if "COMMANDLINE_ARGS" in os.environ
    else ""
)
cmd_opts_dict = vars(cmd_opts)

opts = {}


def get_config():
    if not os.path.exists(cmd_opts.config_file):
        with open(cmd_opts.config_file, "w") as f:
            f.write(toml.dumps(DEFAULT_CONFIG))

    with open(cmd_opts.config_file, mode="r") as f:
        txt = f.read()

    try:
        config = toml.loads(txt)
    except Exception as e:
        print(e)
        config = DEFAULT_CONFIG
    return config


def save_config(options: Dict):
    with open(cmd_opts.config_file, mode="w") as f:
        f.write(toml.dumps(options))


def set(key: str, value: str):
    config = get_config()
    keys = key.split(".")
    if len(keys) == 1:
        config[keys[0]] = value
    else:
        tmp = None
        for k in keys[0:-1]:
            if tmp is None:
                if k not in config:
                    config[k] = {}
                tmp = config[k]
            else:
                if k not in tmp:
                    tmp[k] = {}
                tmp = tmp[k]
        tmp[keys[-1]] = value
    save_config(config)


def get(key: str):
    if key in cmd_opts_dict and cmd_opts_dict[key] is not None:
        return cmd_opts_dict[key]
    config = get_config()
    keys = key.split(".")
    tmp = None
    for k in keys:
        if tmp is None:
            if k not in config:
                return None
            tmp = config[k]
        else:
            if k not in tmp:
                return None
            tmp = tmp[k]
    return tmp


def init():
    global opts
    if not os.path.exists(cmd_opts.config_file):
        save_config(DEFAULT_CONFIG)
    else:
        config = get_config()
        if Version(config["version"]) < Version(DEFAULT_CONFIG["version"]):
            for v in update.update(
                config["version"],
                DEFAULT_CONFIG["version"],
            ):
                config = get_config()
                config["version"] = v
                save_config(config)
