from dataclasses import dataclass, field
from typing import *
from typing import Any

import torch

from api.diffusion.pipelines.diffusers import DiffusersPipelineModel

from . import BaseEvent, SkippableEvent


@dataclass
class LoadResourceEvent(BaseEvent):
    pipe: DiffusersPipelineModel


@dataclass
class PromptTokenizingEvent(BaseEvent):
    pipe: DiffusersPipelineModel
    text_tokens: List
    text_weights: List


@dataclass
class UNetDenoisingEvent(SkippableEvent):
    pipe: DiffusersPipelineModel

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
