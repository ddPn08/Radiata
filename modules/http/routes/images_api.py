import asyncio

from api.models.diffusion import ImageGenerationOptions, ImageGenerationResult
from modules import model_manager

from ..api_router import api
from ..models.base import BaseResponseModel


class GenerateImageResponseModel(BaseResponseModel):
    data: ImageGenerationResult


@api.post("/images/generate", response_model=GenerateImageResponseModel)
async def generate_image(opts: ImageGenerationOptions):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model_manager.runner.generate, opts)
    return GenerateImageResponseModel(
        status="success",
        data=result,
    )
