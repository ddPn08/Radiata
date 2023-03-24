import gc
import random
import time
from itertools import chain
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from api.models.diffusion import (
    DenoiseLatentData,
    ImageGenerationOptions,
    ImageGenerationResult,
)
from modules import config, utils
from modules.app import sio
from modules.diffusion.lpw import LongPromptWeightingPipeline
from modules.images import save_image
from modules.model import StableDiffusionModel

from .runner import BaseRunner


class DiffusersDiffusionRunner(BaseRunner):
    def __init__(self, model: StableDiffusionModel) -> None:
        super().__init__(model)
        self.activate()

    def activate(self) -> None:
        self.loading = True
        self.pipe: Union[
            StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        ] = StableDiffusionPipeline.from_pretrained(
            self.model.model_id,
            use_auth_token=config.get("hf_token"),
            torch_dtype=torch.float16,
            custom_pipeline="lpw_stable_diffusion",
        ).to(
            torch.device("cuda")
        )
        self.pipe.safety_checker = None
        self.pipe.enable_attention_slicing()
        if utils.is_installed("xformers") and config.get("xformers"):
            self.pipe.enable_xformers_memory_efficient_attention()
        self.loading = False

        self.lpw = LongPromptWeightingPipeline(
            self.pipe.text_encoder, self.pipe.tokenizer, self.pipe.device
        )

        def _encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ):
            return self.lpw(prompt, negative_prompt, num_images_per_prompt)

        self.pipe._encode_prompt = _encode_prompt

    def teardown(self) -> None:
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()

    def generate(self, opts: ImageGenerationOptions) -> ImageGenerationResult:
        self.wait_loading()
        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        e2e_tic = time.perf_counter()
        results = []

        for i in range(opts.batch_count):
            manual_seed = opts.seed + i

            def callback(
                step: int,
                timestep: torch.Tensor,
                latents: torch.Tensor,
            ):
                timesteps = self.pipe.scheduler.timesteps
                include = step % 10 == 0 and len(timesteps) - step >= 10
                if include:
                    images = self.pipe.decode_latents(latents)
                    images = self.pipe.numpy_to_pil(images)
                    images = [
                        *images,
                        *list(chain.from_iterable([x for x, _ in results])),
                    ]
                data = DenoiseLatentData(
                    step=step,
                    preview={utils.img2b64(x): opts for x in images} if include else {},
                )

                async def runner():
                    await sio.emit("denoise_latent", data=data.dict())

                utils.fire_and_forget(runner)()

            generator = torch.Generator(device=self.pipe.device).manual_seed(
                manual_seed
            )
            images = self.pipe(
                prompt=opts.prompt,
                negative_prompt=opts.negative_prompt,
                height=opts.image_height,
                width=opts.image_width,
                guidance_scale=opts.scale,
                num_inference_steps=opts.steps,
                generator=generator,
                callback=callback,
            ).images
            results.append(
                (
                    images,
                    ImageGenerationOptions.parse_obj(
                        {"seed": manual_seed, **opts.dict()}
                    ),
                )
            )
            for img in images:
                save_image(img, opts)

        all_perf_time = time.perf_counter() - e2e_tic
        result = ImageGenerationResult(images={}, performance=all_perf_time)
        for images, opts in results:
            for img in images:
                result.images[utils.img2b64(img)] = opts

        return result
