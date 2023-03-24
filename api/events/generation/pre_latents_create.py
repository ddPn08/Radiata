from typing import Optional

import torch

from api.models.diffusion import ImageGenerationOptions

from .base import BaseGenerationEvent


class PreLatentsCreateEvent(BaseGenerationEvent):
    event_name = "pre_latents_create_event"

    def __init__(
        self,
        opts: ImageGenerationOptions,
        latents: Optional[torch.Tensor] = None,
        skip=False,
    ):
        self.opts = opts
        self.latents = latents
        self.skip = skip
