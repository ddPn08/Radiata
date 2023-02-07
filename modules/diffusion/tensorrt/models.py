import torch
from transformers import CLIPTextModel

from lib.trt import models
from submodules.diffusers_0_7_2.src.diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)


class CLIP(models.CLIP):
    def __init__(
        self,
        model_id: str,
        hf_token="",
        text_maxlen=77,
        embedding_dim=768,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
    ):
        super().__init__(
            hf_token, text_maxlen, embedding_dim, fp16, device, verbose, max_batch_size
        )
        self.model_id = model_id

    def get_model(self):
        return CLIPTextModel.from_pretrained(self.model_id).to(self.device)


class UNet(models.UNet):
    def __init__(
        self,
        model_id: str,
        hf_token="",
        text_maxlen=77,
        embedding_dim=768,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
    ):
        super().__init__(
            hf_token, text_maxlen, embedding_dim, fp16, device, verbose, max_batch_size
        )
        self.model_id = model_id

    def get_model(self):
        model_opts = (
            {"revision": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        )
        return UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", use_auth_token=self.hf_token, **model_opts
        ).to(self.device)


class VAE(models.VAE):
    def __init__(
        self,
        model_id: str,
        hf_token="",
        text_maxlen=77,
        embedding_dim=768,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
    ):
        super().__init__(
            hf_token, text_maxlen, embedding_dim, fp16, device, verbose, max_batch_size
        )
        self.model_id = model_id

    def get_model(self):
        vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            use_auth_token=self.hf_token,
        ).to(self.device)
        vae.forward = vae.decode
        return vae
