import gc
import os
from typing import *

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import convert_from_ckpt
from transformers import CLIPTextModel

from modules.shared import ROOT_DIR, hf_diffusers_cache_dir, hf_transformers_cache_dir


def convert_checkpoint_to_pipe(model_id: str):
    ckpt_path = os.path.join(ROOT_DIR, "models", "checkpoints", model_id)
    if os.path.exists(ckpt_path):
        return convert_from_ckpt.download_from_original_stable_diffusion_ckpt(
            ckpt_path,
            from_safetensors=ckpt_path.endswith(".safetensors"),
            load_safety_checker=False,
        )


def load_unet(model_id: str, device: Optional[torch.device] = None):
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        unet = temporary_pipe.unet
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", cache_dir=hf_diffusers_cache_dir()
        )
    unet = unet.to(device=device)
    return unet


def load_text_encoder(model_id: str, device: Optional[torch.device] = None):
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        text_encoder = temporary_pipe.text_encoder
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", cache_dir=hf_transformers_cache_dir()
        )
    text_encoder = text_encoder.to(device=device)
    return text_encoder


def load_vae_decoder(model_id: str, device: Optional[torch.device] = None):
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        vae = temporary_pipe.vae
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", cache_dir=hf_diffusers_cache_dir()
        )

    vae.forward = vae.decode
    vae = vae.to(device=device)
    return vae


def load_vae_encoder(model_id: str, device: Optional[torch.device] = None):
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        vae = temporary_pipe.vae
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", cache_dir=hf_diffusers_cache_dir()
        )

    def encoder_forward(x):
        return vae.encode(x).latent_dist.sample()

    vae.forward = encoder_forward
    vae = vae.to(device=device)
    return vae
