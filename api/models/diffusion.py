from typing import *

from pydantic import BaseModel


class ImageGenerationOptions(BaseModel):
    prompt: str
    negative_prompt: str = ""
    batch_size: int = 1
    batch_count: int = 1
    scheduler_id: str = "euler_a"
    steps: int = 28
    scale: float = 7.5
    image_height: int = 512
    image_width: int = 512
    seed: Optional[int] = None
    strength: Optional[float] = None
    img2img: bool = False


class ImageGenerationError(BaseModel):
    error: Optional[str] = None
    message: str
