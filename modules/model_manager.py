from typing import List, Literal

from modules.logger import logger
from modules.runners.runner_diffusers import DiffusersDiffusionRunner

from . import config
from .model import StableDiffusionModel
from .runners.runner_tensorrt import TensorRTDiffusionRunner

ModelMode = Literal["diffusers", "tensorrt"]

runner = None
mode: ModelMode = config.get("mode")
sd_models: List[StableDiffusionModel] = []
sd_model: StableDiffusionModel = None

raw_model_list = config.get("models") or []
if len(raw_model_list) < 1:
    raw_model_list = config.DEFAULT_CONFIG["models"]
for model_data in raw_model_list:
    sd_models.append(StableDiffusionModel(**model_data))


def set_default_model():
    global sd_model
    prev = config.get("model")
    sd_model = [x for x in sd_models if x.model_id == prev]
    if len(sd_model) != 1:
        sd_model = sd_models[0]
    else:
        sd_model = sd_model[0]

    set_model(sd_model.model_id)


def set_mode(m: ModelMode):
    global mode
    mode = m
    runner.teardown()
    set_model(sd_model.model_id)


def get_model(model_id: str):
    model = [x for x in sd_models if x.model_id == model_id]
    if len(model) < 1:
        return None
    return model[0]


def set_model(model_id: str):
    global runner
    global sd_model
    sd_model = [x for x in sd_models if x.model_id == model_id]
    if len(sd_model) != 1:
        raise ValueError("Model not found or multiple models with same ID.")
    else:
        sd_model = sd_model[0]

    logger.info(f"Loading {sd_model.model_id}...")
    if mode == "diffusers":
        runner = DiffusersDiffusionRunner(sd_model)
    elif mode == "tensorrt":
        runner = TensorRTDiffusionRunner(sd_model)
    logger.info(f"Loaded {sd_model.model_id}...")
