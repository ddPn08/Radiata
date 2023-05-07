import os

import torch

from modules import config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hf_diffusers_cache_dir():
    cache_dir = os.path.join(config.get("model_dir"), "diffusers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def hf_transformers_cache_dir():
    cache_dir = os.path.join(config.get("model_dir"), "transformers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_device():
    device_str = config.get("device")
    if len(device_str.split(",")) > 1:
        device = torch.device(device_str.split(",")[0])
    else:
        device = torch.device(device_str)
    return torch.device(device)
