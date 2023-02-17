from typing import Dict, Literal, Optional

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
    type: Literal["result"] = "result"
    images: Dict[str, ImageInformation]
    performance: float


class ImageGenerationProgress(BaseModel):
    type: Literal["progress"] = "progress"
    progress: float
    performance: float
