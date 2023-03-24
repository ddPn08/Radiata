from typing import List, Optional

from pydantic import BaseModel

from modules import config, model_manager
from modules.model import StableDiffusionModel
from modules.model_manager import ModelMode

from ..api_router import api
from ..models.base import BaseResponseModel


class ModelListResponseModel(BaseResponseModel):
    data: List[StableDiffusionModel] = []


@api.get("/model/list", response_model=ModelListResponseModel)
def get_models():
    return ModelListResponseModel(status="success", data=model_manager.sd_models)


class ModelCurrentResponseModel(BaseResponseModel):
    data: Optional[StableDiffusionModel]


@api.get("/model/currnet", response_model=ModelCurrentResponseModel)
def get_current_model():
    return ModelCurrentResponseModel(status="success", data=model_manager.sd_model)


class SetModelRequest(BaseModel):
    model_id: str


@api.post("/model/current", response_model=BaseResponseModel)
def set_model(req: SetModelRequest):
    model_manager.set_model(req.model_id)
    config.set("model", req.model_id)
    return BaseResponseModel(status="success", message="Set model")


class ModelModeResponseModel(BaseResponseModel):
    data: ModelMode


@api.get("/model/mode", response_model=ModelModeResponseModel)
def get_model_mode():
    return ModelModeResponseModel(status="success", data=model_manager.mode)


class SetModeRequest(BaseModel):
    mode: ModelMode


@api.post("/model/mode", response_model=BaseResponseModel)
def set_model_mode(req: SetModeRequest):
    model_manager.set_mode(req.mode)
    config.set("mode", req.mode)
    return BaseResponseModel(status="success", message="Set mode")


@api.get("/model/{model_id}/trt_available")
def trt_available(model_id: str):
    return model_manager.get_model(model_id).trt_available()
