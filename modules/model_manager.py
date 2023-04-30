from typing import *

from huggingface_hub import HfApi, ModelFilter

from modules.logger import logger
from modules.runners.runner import BaseRunner

from . import config
from .model import DiffusersModel
from .utils import tensorrt_is_available

ModelMode = Literal["diffusers", "tensorrt", "deepfloyd_if"]

runner: Optional[BaseRunner] = None
mode: ModelMode = config.get("mode")
sd_models: List[DiffusersModel] = []
sd_model: Optional[DiffusersModel] = None
available_mode = ["diffusers", "deepfloyd_if"]


def init():
    global mode
    raw_model_list = config.get("models") or []
    if len(raw_model_list) < 1:
        raw_model_list = config.DEFAULT_CONFIG["models"]
    for model_data in raw_model_list:
        sd_models.append(DiffusersModel(**model_data))

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
    prev = mode
    mode = m
    if prev == "deepfloyd_if":
        set_default_model()
        config.set("mode", mode)
        return
    if runner is not None:
        runner.teardown()
    if sd_model is not None:
        set_model(sd_model.model_id)
    config.set("mode", mode)


def get_model(model_id: str):
    model = [x for x in sd_models if x.model_id == model_id]
    if len(model) < 1:
        return None
    return model[0]


def add_model(model_id: str):
    global sd_models
    sd_models.append(DiffusersModel(model_id=model_id))
    config.set("models", [x.dict() for x in sd_models])


def _set_model(model: DiffusersModel):
    global runner
    global sd_model

    sd_model = model

    if runner is not None:
        runner.teardown()
        del runner
        runner = None

    if mode == "diffusers":
        from modules.runners.diffusers import DiffusersDiffusionRunner

        runner = DiffusersDiffusionRunner(sd_model)
    elif mode == "tensorrt":
        from .runners.tensorrt import TensorRTDiffusionRunner

        if not model.trt_available():
            logger.warning("TensorRT is not available for this model.")
            set_default_model()
            return

        runner = TensorRTDiffusionRunner(sd_model)
    elif mode == "deepfloyd_if":
        from .runners.deepfloyd_if import DeepfloydIFRunner

        if (
            sd_model.IF_model_id_1 is None
            or sd_model.IF_model_id_2 is None
            or sd_model.IF_model_id_3 is None
        ):
            sd_model = DiffusersModel(
                model_id="deepfloyd_if",
                IF_model_id_1="DeepFloyd/IF-I-L-v1.0",
                IF_model_id_2="DeepFloyd/IF-II-L-v1.0",
                IF_model_id_3="stabilityai/stable-diffusion-x4-upscaler",
            )

        runner = DeepfloydIFRunner(sd_model)

    config.set("model", sd_model.model_id)


def set_model(model_id: str):
    global runner, sd_model, mode
    sd_model = [x for x in sd_models if x.model_id == model_id]
    if len(sd_model) != 1:
        raise ValueError("Model not found or multiple models with same ID.")
    else:
        sd_model = sd_model[0]

    if mode == "tensorrt" and not sd_model.trt_available():
        logger.warning("TensorRT is not available for this model.")
        mode = "diffusers"
        set_default_model()
        return

    logger.info(f"Loading {sd_model.model_id}...")
    _set_model(sd_model)
    logger.info(f"Loaded {sd_model.model_id}...")


def set_default_model():
    global sd_model, mode
    prev = config.get("model")
    sd_model = [x for x in sd_models if x.model_id == prev]

    if len(sd_model) == 1:
        sd_model = sd_model[0]
    else:
        sd_model = None

    if mode == "tensorrt" and sd_model and not sd_model.trt_available():
        sd_model = None

    if mode == "deepfloyd_if":
        sd_model = DiffusersModel(model_id="deepfloyd_if")
        _set_model(sd_model)
        return

    if sd_model is None:
        available_models = [*sd_models]

        if mode == "tensorrt":
            available_models = [x for x in available_models if x.trt_available()]

        if len(available_models) < 1:
            mode = "diffusers"
            available_models = [*sd_models]

        sd_model = available_models[0]

    set_model(sd_model.model_id)


def search_model(model_id: str):
    api = HfApi()
    models = api.list_models(
        filter=ModelFilter(library="diffusers", model_name=model_id)
    )
    return models
