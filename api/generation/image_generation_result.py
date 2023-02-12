from typing import Dict, Optional

from pydantic import BaseModel


class ImageInformation(BaseModel):
    prompt: str
    negative_prompt: str
    steps: int
    scale: float
    seed: int
    height: int
    width: int
    img2img: bool
    strength: Optional[float]


class ImageGenerationResult(BaseModel):
    images: Dict[str, ImageInformation]
    performance: float
