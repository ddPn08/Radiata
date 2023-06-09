import os
from glob import glob
from typing import *

from huggingface_hub import HfApi, ModelFilter

from modules.logger import logger
from modules.shared import ROOT_DIR

from . import config
from .model import DiffusersModel

sd_models: List[DiffusersModel] = []
sd_model: Optional[DiffusersModel] = None
mode: Literal["stable-diffusion", "deepfloyd_if"] = (
    "deepfloyd_if" if config.get("deepfloyd_if") else "stable-diffusion"
)


def get_model(model_id: str):
    model = [x for x in sd_models if x.model_id == model_id]
    if len(model) < 1:
        return None
    return model[0]


def add_model(model_id: str):
    global sd_models
    sd_models.append(DiffusersModel(model_id=model_id))
    config.set("models", [x.model_id for x in sd_models])


def set_model(model_id: str):
    global sd_model
    sd_model.teardown()

    try:
        sd_model = [x for x in sd_models if x.model_id == model_id]
        if len(sd_model) != 1:
            raise ValueError("Model not found or multiple models with same ID.")
        else:
            sd_model = sd_model[0]

        logger.info(f"Loading {sd_model.model_id}...")
        sd_model.activate()
        config.set("model", sd_model.model_id)
        logger.info(f"Loaded {sd_model.model_id}...")
    except Exception as e:
        logger.error(f"Failed to load {model_id}...")
        logger.error(e)
        set_default_model()


def search_model(model_id: str):
    api = HfApi()
    models = api.list_models(filter=ModelFilter(model_name=model_id))
    return list(iter(models))


def set_default_model():
    global sd_model
    prev = config.get("model")
    sd_model = [x for x in sd_models if x.model_id == prev]

    if len(sd_model) == 1:
        sd_model = sd_model[0]
    else:
        sd_model = [*sd_models][0]

    set_model(sd_model.model_id)


def reload_models():
    sd_models.clear()
    raw_model_list = config.get("models") or []
    if len(raw_model_list) < 1:
        raw_model_list = config.DEFAULT_CONFIG["models"]
    for model_id in raw_model_list:
        sd_models.append(DiffusersModel(model_id=model_id))

    checkpoints_path = os.path.join(ROOT_DIR, "models", "checkpoints")

    for model in glob(os.path.join(checkpoints_path, "**", "*"), recursive=True):
        if model.endswith(".safetensors") or model.endswith(".ckpt"):
            relpath = os.path.relpath(model, checkpoints_path)
            model_id = relpath.replace(os.sep, "/")
            if model_id not in raw_model_list:
                sd_models.append(DiffusersModel(model_id=model_id))


def init():
    if mode != "stable-diffusion":
        return
    reload_models()
    set_default_model()
