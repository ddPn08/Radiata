from typing import *

from huggingface_hub import HfApi, ModelFilter

from modules.logger import logger

from . import config
from .model import StableDiffusionModel
from .utils import tensorrt_is_available

ModelMode = Literal["diffusers", "tensorrt"]

runner = None
mode: ModelMode = config.get("mode")
sd_models: List[StableDiffusionModel] = []
sd_model: Optional[StableDiffusionModel] = None
available_mode = ["diffusers"]


def init():
    global mode
    raw_model_list = config.get("models") or []
    if len(raw_model_list) < 1:
        raw_model_list = config.DEFAULT_CONFIG["models"]
    for model_data in raw_model_list:
        sd_models.append(StableDiffusionModel(**model_data))

    trt_module_status, trt_version_status = tensorrt_is_available()
    if config.get("tensorrt"):
        if trt_module_status and trt_version_status:
            available_mode.append("tensorrt")
        elif trt_module_status:
            logger.warning(
                "TensorRT is available, but torch version is not compatible."
            )

    if mode not in available_mode:
        mode = available_mode[0]
        config.set("mode", mode)


def set_mode(m: ModelMode):
    global mode
    if m == mode:
        return
    mode = m
    runner.teardown()
    set_model(sd_model.model_id)
    config.set("mode", mode)


def get_model(model_id: str):
    model = [x for x in sd_models if x.model_id == model_id]
    if len(model) < 1:
        return None
    return model[0]


def add_model(model_id: str):
    global sd_models
    sd_models.append(StableDiffusionModel(model_id=model_id))
    config.set("models", [x.dict() for x in sd_models])


def set_model(model_id: str):
    global runner
    global sd_model
    sd_model = [x for x in sd_models if x.model_id == model_id]
    if len(sd_model) != 1:
        raise ValueError("Model not found or multiple models with same ID.")
    else:
        sd_model = sd_model[0]

    if runner is not None:
        runner.teardown()
        del runner

    logger.info(f"Loading {sd_model.model_id}...")
    if mode == "diffusers":
        from modules.runners.diffusers import DiffusersDiffusionRunner

        runner = DiffusersDiffusionRunner(sd_model)
    elif mode == "tensorrt":
        from .runners.tensorrt import TensorRTDiffusionRunner

        runner = TensorRTDiffusionRunner(sd_model)
    logger.info(f"Loaded {sd_model.model_id}...")

    config.set("model", sd_model.model_id)


def set_default_model():
    global sd_model
    prev = config.get("model")
    sd_model = [x for x in sd_models if x.model_id == prev]

    if len(sd_model) == 1:
        sd_model = sd_model[0]

    if mode == "tensorrt" and not sd_model.trt_available():
        sd_model = None

    if sd_model is None:
        available_models = [*sd_models]

        if mode == "tensorrt":
            available_models = [x for x in available_models if x.trt_available()]

        if len(available_models) < 1:
            set_mode("diffusers")

        sd_model = available_models[0]

    set_model(sd_model.model_id)


def search_model(model_id: str):
    api = HfApi()
    models = api.list_models(
        filter=ModelFilter(library="diffusers", model_name=model_id)
    )
    return models
