import io
from typing import Optional

from fastapi import Response
from pydantic import BaseModel

from api.generation import ImageGenerationOptions, ImageGenerationResult
from modules import images, runners

from ..api_router import api
from ..models.base import BaseResponseModel


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    batch_size: int = 1
    batch_count: int = 1
    scheduler_id: str = "euler_a"
    steps: int = 28
    scale: int = 7.5
    image_height: int = 512
    image_width: int = 512
    seed: Optional[int] = None
    strength: Optional[float] = None
    img: Optional[str] = None


class GenerateImageResponseModel(BaseResponseModel):
    data: ImageGenerationResult


@api.post("/images/generate", response_model=GenerateImageResponseModel)
def generate_image(req: ImageGenerationOptions):
    result = runners.generate(req)
    return GenerateImageResponseModel(
        status="success",
        data=result,
    )


@api.get("/images/{category}/{filename}")
def get_image(category: str, filename: str):
    byteio = io.BytesIO()
    img = images.get_image(category, filename)
    img.save(byteio, format="PNG")
    return Response(content=byteio.getvalue(), media_type="image/png")
