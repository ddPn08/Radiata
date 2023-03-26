#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/stable_diffusion_pipeline.py

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import diffusers
import torch
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from tqdm import tqdm
from transformers import CLIPTokenizer

from .models import BaseModel
from .utilities import (
    Engine,
    create_models,
    decode_images,
    device_view,
    preprocess_image,
)


class TensorRTDiffusionPipeline:
    stages = []

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        onnx_dir: str,
        engine_dir: str,
        use_auth_token: Optional[str],
        device: Union[str, torch.device],
        onnx_refit_dir: Optional[str] = None,
        max_batch_size: int = 1,
    ):
        models = create_models(
            model_id=model_id,
            stages=cls.stages,
            use_auth_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
        )

        engines = {}

        for model_name in models.keys():
            engine_path = os.path.join(engine_dir, model_name + ".plan")
            assert os.path.exists(
                engine_path
            ), f"TensorRT engine for {model_id} is not built."
            engine = Engine(engine_path)
            engines[model_name] = engine

        # Load and activate TensorRT engines
        for model_name in models.keys():
            engine = engines[model_name]
            engine.load()
            if onnx_refit_dir:
                onnx_opt_path = os.path.join(onnx_dir, model_name + ".opt.onnx")
                onnx_refit_path = os.path.join(model_name, onnx_refit_dir + ".opt.onnx")
                if os.path.exists(onnx_refit_path):
                    engine.refit(onnx_opt_path, onnx_refit_path)
            engine.activate()

        pipe = cls(
            model_id=model_id,
            engine=engines,
            models=models,
            max_batch_size=max_batch_size,
            device=device,
        )

        pipe.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer", use_auth_token=use_auth_token
        )
        pipe.scheduler = diffusers.DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", use_auth_token=use_auth_token
        )

        return pipe

    def __init__(
        self,
        models: Dict[str, BaseModel],
        engine: Dict[str, Engine],
        model_id: str,
        max_batch_size: int = 1,
        device: torch.device = torch.device("cuda"),
    ):
        self.tokenizer: CLIPTokenizer = None
        self.scheduler: diffusers.DDIMScheduler = None
        self.generator: torch.Generator = None

        self.models = models
        self.engine = engine
        self.model_id = model_id
        self.max_batch_size = max_batch_size
        self.device = device

        self.stream = cuda.Stream()
        self.events = {}

    def __del__(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        for engine in self.engine.values():
            del engine

        self.stream.free()
        del self.stream

    def load_resources(
        self,
        image_height: int,
        image_width: int,
        batch_size: int,
        num_inference_steps: int,
        device: torch.device,
    ):
        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width),
                device=device,
            )

        # Create CUDA events and stream
        for stage in ["clip", "denoise", "vae", "vae_encoder"]:
            for marker in ["start", "stop"]:
                self.events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

    def run_engine(self, model_name: str, feed_dict: Dict[str, cuda.DeviceView]):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def initialize_latents(
        self,
        batch_size: int,
        unet_channels: int,
        height: torch.Tensor,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[torch.Generator, List[torch.Generator]],
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        shape = (batch_size, unet_channels, height, width)
        latents = diffusers.utils.randn_tensor(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def initialize_timesteps(
        self, num_inference_steps: int, strength: int, device: torch.device
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        offset = (
            self.scheduler.steps_offset
            if hasattr(self.scheduler, "steps_offset")
            else 0
        )
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        num_inference_steps = self.scheduler.timesteps[t_start:].to(device)
        return num_inference_steps, t_start

    def prepare_extra_step_kwargs(self, generator: torch.Generator, eta: float):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def preprocess_images(
        self, batch_size: int, device: torch.device, images: Tuple[Image.Image] = ()
    ):
        init_images: List[torch.Tensor] = []
        for image in images:
            if isinstance(image, torch.Tensor):
                init_images.append(image)
                continue
            image = preprocess_image(image)
            image = image.to(device).float()
            image = image.repeat(batch_size, 1, 1, 1)
            init_images.append(image)
        return tuple(init_images)

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Union[str, torch.device],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Union[str, List[str]],
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        cudart.cudaEventRecord(self.events["clip-start"], 0)

        # Tokenize prompt
        text_input_ids = (
            self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(device)
        )

        text_input_ids_inp = device_view(text_input_ids)
        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings: torch.Tensor = self.run_engine(
            "clip", {"input_ids": text_input_ids_inp}
        )["text_embeddings"].clone()
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # Tokenize negative prompt
        uncond_input_ids = (
            self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(device)
        )
        uncond_input_ids_inp = device_view(uncond_input_ids)
        uncond_embeddings = self.run_engine(
            "clip", {"input_ids": uncond_input_ids_inp}
        )["text_embeddings"]

        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(
            dtype=torch.float16
        )

        cudart.cudaEventRecord(self.events["clip-stop"], 0)

        return text_embeddings

    def denoise_latent(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps=None,
        step_offset=0,
        guidance_scale: int = 7.5,
        mask: Optional[torch.Tensor] = None,
        masked_image_latents: Optional[torch.Tensor] = None,
        extra_step_kwargs: dict = {},
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        cudart.cudaEventRecord(self.events["denoise-start"], 0)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step, timestep in enumerate(tqdm(timesteps)):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            # latent_model_input = self.scheduler.scale_model_input(
            #     latent_model_input, step_offset + step_index, timestep
            # )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep
            )

            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat(
                    [latent_model_input, mask, masked_image_latents], dim=1
                )

            # Predict the noise residual
            timestep_float = (
                timestep.float() if timestep.dtype != torch.float32 else timestep
            )

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            noise_pred = self.run_engine(
                "unet",
                {
                    "sample": sample_inp,
                    "timestep": timestep_inp,
                    "encoder_hidden_states": embeddings_inp,
                },
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=latents,
                **extra_step_kwargs,
            ).prev_sample

            if step == len(timesteps) - 1 or (step + 1) % self.scheduler.order == 0:
                if callback is not None and step % callback_steps == 0:
                    callback(step, timestep, latents)

        latents = 1.0 / 0.18215 * latents
        cudart.cudaEventRecord(self.events["denoise-stop"], 0)
        return latents

    def encode_image(self, init_image):
        cudart.cudaEventRecord(self.events["vae_encoder-start"], 0)
        init_latents: torch.Tensor = self.run_engine(
            "vae_encoder", {"images": device_view(init_image)}
        )["latent"]
        cudart.cudaEventRecord(self.events["vae_encoder-stop"], 0)

        init_latents = 0.18215 * init_latents
        return init_latents

    def decode_latent(self, latents):
        cudart.cudaEventRecord(self.events["vae-start"], 0)
        images = self.run_engine("vae", {"latent": device_view(latents)})["images"]
        cudart.cudaEventRecord(self.events["vae-stop"], 0)
        return images

    def decode_images(self, images: torch.Tensor):
        return decode_images(images)
