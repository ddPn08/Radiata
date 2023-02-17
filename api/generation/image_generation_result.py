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


class BaseModelStream(BaseModel):
    def ndjson(self):
        return self.json() + '\n'

class ImageGenerationResult(BaseModelStream):
    type: Literal["result"] = "result"
    images: Dict[str, ImageInformation]
    performance: float

class ImageGenerationProgress(BaseModelStream):
    type: Literal["progress"] = "progress"
    progress: float
    performance: float

class ImageGenerationResult(BaseModelStream):
    type: Literal["result"] = "result"
    images: Dict[str, ImageInformation]
    performance: float


class BuildEngineResult(BaseModelStream):
    message: str
    progress: float