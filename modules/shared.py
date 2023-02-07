import argparse
import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser()


parser.add_argument("--allow_hosts", type=str, default="")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--config_file", type=str, default="config.json")

cmd_opts, _ = parser.parse_known_args(
    os.environ["COMMANDLINE_ARGS"].split(" ") if "COMMANDLINE_ARGS" in os.environ else ""
)
