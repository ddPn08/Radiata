from dataclasses import dataclass, field
from typing import *

import torch

from . import BaseEvent, SkippableEvent


@dataclass
class LoadResourceEvent(BaseEvent):
    pipe: Any


@dataclass
class UNetDenoisingEvent(SkippableEvent):
    pipe: Any

    latent_model_input: torch.Tensor
    step: int
    timestep: torch.Tensor

    latents: torch.Tensor
    timesteps: torch.Tensor
    do_classifier_free_guidance: bool
    prompt_embeds: torch.Tensor
    extra_step_kwargs: Dict[str, Any]
    callback: Optional[Callable]
    callback_steps: int
    cross_attention_kwargs: Dict[str, Any]

    unet_additional_kwargs: Dict[str, Any] = field(default_factory=dict)
