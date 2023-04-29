#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/txt2img_pipeline.py

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

from .pipeline import TensorRTDiffusionPipeline
from .utilities import TRT_LOGGER


class TensorRTStableDiffusionPipeline(TensorRTDiffusionPipeline):
    stages = ["clip", "denoise", "vae"]

    def __init__(self, *args, **kwargs):
        super(TensorRTStableDiffusionPipeline, self).__init__(*args, **kwargs)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image_height: int = 512,
        image_width: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        negative_prompt: Union[str, List[str]] = "",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
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
            device=self.device,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator=generator, eta=eta)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # CLIP text encoder
            text_embeddings = self._encode_prompt(
                prompt, self.device, num_images_per_prompt, True, negative_prompt
            )

            # Pre-initialize latents
            latents = self.initialize_latents(
                batch_size=batch_size * num_images_per_prompt,
                unet_channels=4,
                height=(image_height // 8),
                width=(image_width // 8),
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )

            torch.cuda.synchronize()

            # UNet denoiser
            latents = self.denoise_latent(
                latents=latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
                callback=callback,
                callback_steps=callback_steps,
            )

            # VAE decode latent
            images = self.decode_latent(latents)

            torch.cuda.synchronize()

            return self.decode_images(images)
