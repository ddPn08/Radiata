from typing import Dict, Literal, Optional

from pydantic import BaseModel
from ..base import BaseModelStream


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

class ImageGenerationResult(BaseModelStream):
    type: Literal["result"] = "result"
    images: Dict[str, ImageInformation]
    performance: float

class ImageGenerationError(BaseModelStream):
    type: Literal["error"] = "error"
    error: Optional[str] = None
    message: str

class ImageGenerationProgress(BaseModelStream):
    type: Literal["progress"] = "progress"
    images: Dict[str, ImageInformation]
    progress: float
    performance: float
