import argparse
import json
import os

DEFAULT_CONFIG = {
    "images/txt2img/save_dir": "outputs/txt2img",
    "images/txt2img/save_name": "{index}-{seed}-{prompt}.png",
    "images/img2img/save_dir": "outputs/img2img",
    "images/img2img/save_name": "{index}-{seed}-{prompt}.png",
}


parser = argparse.ArgumentParser()


parser.add_argument("--allow-hosts", type=str, default="")
parser.add_argument("--model-dir", type=str, default="models")
parser.add_argument("--config-file", type=str, default="config.json")
parser.add_argument("--hf-token", type=str)

cmd_opts, _ = parser.parse_known_args(
    os.environ["COMMANDLINE_ARGS"].split(" ")
    if "COMMANDLINE_ARGS" in os.environ
    else ""
)
cmd_opts_dict = vars(cmd_opts)

opts = {}


def get_config():
    with open(cmd_opts.config_file, mode="r") as f:
        txt = f.read()
        return json.loads(txt)


def save_config():
    with open(cmd_opts.config_file, mode="w") as f:
        f.write(json.dumps(opts))


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
        opts = get_config()
