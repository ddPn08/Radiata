import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser()


parser.add_argument("--dev", action="store_true")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--config_file", type=str, default="config.json")

cmd_opts, _ = parser.parse_known_args(sys.argv)
