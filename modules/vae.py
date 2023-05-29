import os

from modules.shared import ROOT_DIR


def list_vae_models():
    dir = os.path.join(ROOT_DIR, "models", "vae")
    os.makedirs(dir, exist_ok=True)
    return os.listdir(dir)


def resolve_vae(name: str):
    return os.path.join(ROOT_DIR, "models", "vae", name)
