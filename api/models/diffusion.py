import json
from dataclasses import asdict, dataclass, field
from typing import *

import PIL.Image


@dataclass
class HiresfixOptions:
    enable: bool = False
    mode: str = "bilinear"
    scale: float = 1.5


@dataclass
class MultidiffusionOptions:
    enable: bool = False
    views_batch_size: int = 1
    window_size: int = 64
    stride: int = 8


@dataclass
class ImageGenerationOptions:
    # serializable
    prompt: str
    negative_prompt: str = ""
    batch_size: int = 1
    batch_count: int = 1
    scheduler_id: str = "euler_a"
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    strength: Optional[float] = 1.0

    image: PIL.Image.Image = field(default_factory=PIL.Image.Image)

    hiresfix: HiresfixOptions = None
    multidiffusion: MultidiffusionOptions = None

    def dict(self):
        return asdict(self)

    def json(self):
        d = self.dict()
        del d["image"]
        return json.dumps(d)

    @classmethod
    def parse_obj(cls, obj):
        return ImageGenerationOptions(**obj)
