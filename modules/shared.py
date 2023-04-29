import os

from modules import config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hf_cache_dir():
    cache_dir = os.path.join(config.get("model_dir"), "diffusers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
