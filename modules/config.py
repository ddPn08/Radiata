import json
import os

from modules import shared

DEFAULT_CONFIG = {
    "images/txt2img/save_dir": "outputs/txt2img",
    "images/txt2img/save_name": "{seed}-{prompt}.png",
}

opts = {}


def get_config():
    with open(shared.cmd_opts.config_file, mode="r") as f:
        txt = f.read()
        return json.loads(txt)


def save_config():
    with open(shared.cmd_opts.config_file, mode="w") as f:
        f.write(json.dumps(opts))


def set(key: str, value: str):
    opts[key] = value
    save_config()


def get(key: str):
    config = get_config()
    return config[key] if key in config else None


def init():
    global opts
    if not os.path.exists(shared.cmd_opts.config_file):
        opts = DEFAULT_CONFIG
        save_config()
    else:
        opts = get_config()
