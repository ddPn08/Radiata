import argparse
import os

import toml
from packaging.version import Version

from .version import update

DEFAULT_CONFIG = {
    "version": "0.0.1",
    "images/txt2img/save_dir": "outputs/txt2img",
    "images/txt2img/save_name": "{index}-{seed}-{prompt}.png",
    "images/img2img/save_dir": "outputs/img2img",
    "images/img2img/save_name": "{index}-{seed}-{prompt}.png",
    "model_dir": "models",
    "models": ["runwayml/stable-diffusion-v1-5"],
    "model": "runwayml/stable-diffusion-v1-5",
    "mode": "diffusers",
}

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


parser = argparse.ArgumentParser()

parser.add_argument("--config-file", type=str, default="config.toml")

parser.add_argument("--host", type=str, default="")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")

parser.add_argument("--model-dir", type=str, default="models")
parser.add_argument("--hf-token", type=str)

parser.add_argument("--xformers", action="store_true")
parser.add_argument("--tensorrt", action="store_true")

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


def save_config(options: dict = None):
    with open(cmd_opts.config_file, mode="w") as f:
        f.write(toml.dumps(options or opts))


def set(key: str, value: str):
    opts[key] = value
    save_config()


def get(key: str):
    if key in cmd_opts_dict and cmd_opts_dict[key] is not None:
        return cmd_opts_dict[key]
    config = get_config()
    return (
        config[key]
        if key in config
        else (DEFAULT_CONFIG[key] if key in DEFAULT_CONFIG else None)
    )


def init():
    global opts
    if not os.path.exists(cmd_opts.config_file):
        opts = DEFAULT_CONFIG
        save_config()
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

        opts = get_config()
