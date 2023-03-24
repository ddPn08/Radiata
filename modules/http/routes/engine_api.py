from api.models.tensorrt import BuildEngineOptions
from modules import model_manager
from modules.http.models.base import BaseResponseModel

from ...acceleration.tensorrt.engine import EngineBuilder
from ..api_router import api


@api.post("/engine/build")
async def build_engine(req: BuildEngineOptions):
    model_manager.runner.teardown()
    builder = EngineBuilder(req)
    builder.build()
    model_manager.runner.activate()

    return BaseResponseModel(status="success", message="Finish build engine")
