from typing import List
from typing import Optional
from pydantic import BaseModel

from modules import runner
from modules.api.models.base import BaseResponseModel

from ..api_router import api


class ModelListResponseModel(BaseResponseModel):
    data: List[str] = []


@api.get("/model/list", response_model=ModelListResponseModel)
def get_runners():
    data = runner.get_runners()
    return ModelListResponseModel(status="success", data=data)


class ModelCurrentResponseModel(BaseResponseModel):
    data: Optional[str]


@api.get("/model/currnet", response_model=ModelCurrentResponseModel)
def get_current_runner():
    return ModelCurrentResponseModel(
        status="success",
        data=runner.current.model_id if runner.current is not None else None,
    )


class SetRunnerRequest(BaseModel):
    model_id: str
    tokenizer_id: str = "openai/clip-vit-large-patch14"


@api.post("/model/set", response_model=BaseResponseModel)
def set_runner(req: SetRunnerRequest):
    runner.set_runner(req.model_id, req.tokenizer_id)
    return BaseResponseModel(status="success", message="Set model")
