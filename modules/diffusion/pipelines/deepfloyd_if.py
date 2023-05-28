import gc
from typing import *

import psutil
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from transformers import T5EncoderModel

from modules import config
from modules.shared import get_device, hf_diffusers_cache_dir, hf_transformers_cache_dir


class IFDiffusionPipeline:
    @classmethod
    def from_pretrained(
        cls, model_id_I: str, model_id_II: str, model_id_III: str, mode: str = "auto"
    ):
        device = get_device()
        if device.type == "cuda":
            vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            vram = 0
        ram = psutil.virtual_memory().total / (1024**3)
        if mode == "auto":
            if vram <= 6:
                if ram > 32:
                    mode = "sequential_off_load"
            elif vram <= 16:
                if ram > 16:
                    mode = "off_load"
                else:
                    mode = "lowvram"
            elif vram <= 24:
                mode = "medvram"
                if ram > 16:
                    mode = "off_load"
            else:
                mode = "normal"

        return cls(
            mode=mode,
            IF_I_id=model_id_I,
            IF_II_id=model_id_II,
            IF_III_id=model_id_III,
            torch_dtype=torch.float16 if config.get("fp16") else torch.float32,
        )

    def __init__(
        self,
        mode: Literal[
            "lowvram", "sequential_off_load", "off_load", "medvram", "normal"
        ] = "normal",
        IF_I_id: str = None,
        IF_II_id: str = None,
        IF_III_id: str = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.mode = mode
        self.IF_I_id = IF_I_id
        self.IF_II_id = IF_II_id
        self.IF_III_id = IF_III_id

        self.torch_dtype = torch_dtype
        self.variant = "fp16" if torch_dtype == torch.float16 else None

        device_str = config.get("device")
        if len(device_str.split(",")) == 4:
            self.device = [torch.device(d) for d in device_str.split(",")]
        else:
            self.device = [torch.device(device_str)] * 4

        self.t5 = None
        self.IF_I = None
        self.IF_II = None
        self.IF_III = None
        self.previous = {
            "prompt_embeds": None,
            "negative_prompt_embeds": None,
            "images_I": None,
            "images_II": None,
        }

    def _flush(self):
        gc.collect()
        torch.cuda.empty_cache()

    def load_pipeline(
        self,
        stage: Literal["I", "II", "III"],
        pipe_type: Literal["t5", "IF_I", "IF_II", "IF_III"],
        **kwargs
    ):
        if stage == "I":
            if self.mode == "lowvram" or self.mode == "medvram":
                self.IF_II = None
                self.IF_III = None
        elif stage == "II":
            if self.mode == "lowvram" or self.mode == "medvram":
                self.t5 = None
                self.IF_I = None
                self.IF_III = None
        elif stage == "III":
            if self.mode == "lowvram" or self.mode == "medvram":
                self.t5 = None
                self.IF_I = None
                self.IF_II = None

        self._flush()

        if pipe_type == "t5":
            if self.t5 is None:
                self.t5 = T5EncoderModel.from_pretrained(
                    "DeepFloyd/IF-I-XL-v1.0",
                    subfolder="text_encoder",
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    cache_dir=hf_transformers_cache_dir(),
                    **kwargs,
                ).to(self.device[0])
        elif pipe_type == "IF_I":
            if self.IF_I is None:
                self.IF_I = DiffusionPipeline.from_pretrained(
                    self.IF_I_id,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    cache_dir=hf_diffusers_cache_dir(),
                    **kwargs,
                ).to(self.device[1])
                if self.mode == "off_load":
                    self.IF_I.enable_model_cpu_offload()
                elif self.mode == "sequential_off_load":
                    self.IF_I.enable_sequential_cpu_offload()
        elif pipe_type == "IF_II":
            if self.IF_II is None:
                self.IF_II = DiffusionPipeline.from_pretrained(
                    self.IF_II_id,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    cache_dir=hf_diffusers_cache_dir(),
                    **kwargs,
                ).to(self.device[2])
                if self.mode == "off_load":
                    self.IF_II.enable_model_cpu_offload()
                elif self.mode == "sequential_off_load":
                    self.IF_II.enable_sequential_cpu_offload()
        elif pipe_type == "IF_III":
            if self.IF_III is None:
                self.IF_III = DiffusionPipeline.from_pretrained(
                    self.IF_III_id,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    cache_dir=hf_diffusers_cache_dir(),
                    **kwargs,
                ).to(self.device[3])
                if self.mode == "off_load":
                    self.IF_III.enable_model_cpu_offload()
                elif self.mode == "sequential_off_load":
                    self.IF_III.enable_sequential_cpu_offload()

    def _encode_prompt(self, prompt: str, negative_prompt: str):
        self.load_pipeline("I", "t5")
        if self.mode == "lowvram":
            self.load_pipeline("I", "IF_I", text_encoder=self.t5, unet=None)
        else:
            self.load_pipeline("I", "IF_I", text_encoder=self.t5)
        prompt_embeds, negative_embeds = self.IF_I.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt
        )
        if self.mode == "lowvram":
            self.t5 = None
            self._flush()

        return prompt_embeds, negative_embeds

    def stage_I(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
    ):
        self.previous = {
            "prompt_embeds": None,
            "negative_prompt_embeds": None,
            "images_I": None,
            "images_II": None,
        }
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt, negative_prompt
        )

        self.load_pipeline("I", "IF_I", text_encoder=None)
        images = self.IF_I(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pt",
        ).images

        self.previous["prompt_embeds"] = prompt_embeds
        self.previous["negative_prompt_embeds"] = negative_prompt_embeds
        self.previous["images_I"] = images

        yield [(pt_to_pil(images), {})]

    def stage_II(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
    ):
        self.load_pipeline("II", "IF_II", text_encoder=None)

        images = self.IF_II(
            image=self.previous["images_I"],
            prompt_embeds=self.previous["prompt_embeds"],
            negative_prompt_embeds=self.previous["negative_prompt_embeds"],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pt",
        ).images

        self.previous["images_II"] = images

        yield [(pt_to_pil(self.previous["images_I"]), {}), (pt_to_pil(images), {})]

    def stage_III(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
    ):
        self.load_pipeline("II", "IF_III")

        images = self.IF_III(
            image=self.previous["images_II"],
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images

        yield [
            (pt_to_pil(self.previous["images_I"]), {}),
            (pt_to_pil(self.previous["images_II"]), {}),
            (images, {}),
        ]
