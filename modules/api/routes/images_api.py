import io

from fastapi import Response
from pydantic import BaseModel

from modules import images, runner
from modules.api.models.base import BaseResponseModel

from ..api_router import api


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
    seed: int | None = None


class GenerateImageResponseData(BaseModel):
    images: dict
    performance: float


class GenerateImageResponseModel(BaseResponseModel):
    data: GenerateImageResponseData


@api.post("/images/generate", response_model=GenerateImageResponseModel)
def generate_image(req: GenerateImageRequest):
    images, performance = runner.generate(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size,
        batch_count=req.batch_count,
        scheduler_id=req.scheduler_id,
        steps=req.steps,
        scale=req.scale,
        seed=None if req.seed == -1 else req.seed,
        image_height=req.image_height,
        image_width=req.image_width,
    )
    return GenerateImageResponseModel(
        status="success",
        data=GenerateImageResponseData(images=images, performance=performance),
    )


@api.get("/images/{category}/{filename}")
def get_image(category: str, filename: str):
    byteio = io.BytesIO()
    img = images.get_image(category, filename)
    img.save(byteio, format="PNG")
    return Response(content=byteio.getvalue(), media_type="image/png")
