#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/img2img_pipeline.py

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

from typing import *

import tensorrt as trt
import torch
from PIL import Image

from .pipeline import TensorRTDiffusionPipeline
from .utilities import TRT_LOGGER


class TensorRTStableDiffusionImg2ImgPipeline(TensorRTDiffusionPipeline):
    stages = ["clip", "denoise", "vae", "vae_encoder"]

    def __init__(self, scheduler="DDIM", *args, **kwargs):
        """
        Initializes the Img2Img Diffusion pipeline.
        """
        super(TensorRTStableDiffusionImg2ImgPipeline, self).__init__(
            *args,
            **kwargs,
            scheduler=scheduler,
            stages=["vae_encoder", "clip", "unet", "vae"]
        )

    def infer(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, Image.Image],
        image_height: int = 512,
        image_width: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        strength: int = 0.75,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            init_image (image):
                Input image to be used as input.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            seed (int):
                Seed for the random generator
            strength (float):
                How much to transform the input image. Must be between 0 and 1
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
        """
        if type(prompt) != list:
            prompt = [prompt]
        if type(negative_prompt) != list:
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(negative_prompt)

        batch_size = len(prompt)

        self.load_resources(
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()

            # Initialize timesteps
            timesteps, t_start = self.initialize_timesteps(
                num_inference_steps, strength
            )
            latent_timestep = timesteps[:1].repeat(batch_size)

            # Pre-process input image
            image = self.preprocess_images(batch_size, (image,))[0]

            # VAE encode init image
            init_latents = self.encode_image(image)

            # CLIP text encoder
            text_embeddings = self._encode_prompt(prompt, negative_prompt)

            # Add noise to latents using timesteps
            noise = torch.randn(
                init_latents.shape,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )

            latents = self.scheduler.add_noise(
                init_latents, noise, t_start, latent_timestep
            )

            # UNet denoiser
            latents = self.denoise_latent(
                latents,
                text_embeddings,
                timesteps=timesteps,
                step_offset=t_start,
                guidance_scale=guidance_scale,
                callback=callback,
                callback_steps=callback_steps,
            )

            # VAE decode latent
            images = self.decode_latent(latents)

            torch.cuda.synchronize()

            return self.decode_images(images)
