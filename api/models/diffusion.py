from typing import Dict, List, Optional

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
    img: Optional[str] = None


class ImageGenerationResult(BaseModel):
    images: Dict[str, ImageGenerationOptions]
    performance: float


class ImageGenerationError(BaseModel):
    error: Optional[str] = None
    message: str


class ImageGenerationProgress(BaseModel):
    images: Dict[str, ImageGenerationOptions]
    progress: float
    performance: float


class DenoiseLatentData(BaseModel):
    step: int
    preview: Optional[Dict[str, ImageGenerationOptions]]
