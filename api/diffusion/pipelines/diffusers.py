import inspect
from dataclasses import dataclass
from typing import *

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from api.models.diffusion import ImageGenerationOptions
from api.plugin import get_plugin_id


@dataclass
class PipeSession:
    plugin_data: Dict[str, Any]
    opts: ImageGenerationOptions


class DiffusersPipelineModel:
    __mode__ = "diffusers"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_id: str,
        use_auth_token: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        subfolder: Optional[str] = None,
    ):
        pass

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.unet: UNet2DConditionModel
        self.scheduler: DDPMScheduler
        self.device: torch.device
        self.dtype: torch.dtype
        self.session: PipeSession
        pass

    def get_plugin_data(self):
        id = get_plugin_id(inspect.stack()[1])
        return self.session.plugin_data[id]

    def set_plugin_data(self, data):
        id = get_plugin_id(inspect.stack()[1])
        self.session.plugin_data[id] = data

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        pass

    def enterers(self):
        pass

    def load_resources(
        self,
        image_height: int,
        image_width: int,
        batch_size: int,
        num_inference_steps: int,
    ):
        pass

    def get_timesteps(self, num_inference_steps: int, strength: Optional[float]):
        pass

    def get_timesteps(self, num_inference_steps: int, strength: Optional[float]):
        pass

    def prepare_extra_step_kwargs(self, generator: torch.Generator, eta):
        pass

    def preprocess_image(self, image: PIL.Image.Image, height: int, width: int):
        pass

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        pass

    def prepare_latents(
        self,
        vae_scale_factor: int,
        unet_in_channels: int,
        image: Optional[torch.Tensor],
        timestep: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: torch.Generator,
        latents: torch.Tensor = None,
    ):
        pass

    def denoise_latent(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
        prompt_embeds: torch.Tensor,
        extra_step_kwargs: Dict[str, Any],
        callback: Optional[Callable],
        callback_steps: int,
        cross_attention_kwargs: Dict[str, Any],
    ):
        pass

    def decode_latents(self, latents: torch.Tensor):
        pass

    def decode_images(self, image: np.ndarray):
        pass

    def create_output(self, latents: torch.Tensor, output_type: str, return_dict: bool):
        pass

    def __call__(
        self,
        opts: ImageGenerationOptions,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        plugin_data: Optional[Dict[str, Any]] = {},
    ):
        pass

    def enable_xformers_memory_efficient_attention(
        self, attention_op: Optional[Callable] = None
    ):
        pass

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        pass
