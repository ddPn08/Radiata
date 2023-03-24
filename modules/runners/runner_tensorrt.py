import gc
import os
import random
import time
from itertools import chain
from typing import Optional, Union

import numpy as np
import torch
from polygraphy import cuda

from api.models.diffusion import (
    DenoiseLatentData,
    ImageGenerationOptions,
    ImageGenerationResult,
)
from lib.tensorrt.pipeline_stable_diffusion import TensorRTStableDiffusionPipeline
from lib.tensorrt.pipeline_stable_diffusion_img2img import (
    TensorRTStableDiffusionImg2ImgPipeline,
)
from modules import config, utils
from modules.acceleration.tensorrt.text_encoder import TensorRTCLIPTextModel
from modules.app import sio
from modules.diffusion.lpw import LongPromptWeightingPipeline
from modules.images import save_image
from modules.model import StableDiffusionModel

from .runner import BaseRunner


class TensorRTDiffusionRunner(BaseRunner):
    def __init__(self, model: StableDiffusionModel) -> None:
        super().__init__(model)

        model_dir = model.get_trt_path()

        self.engine_dir = os.path.join(model_dir, "engine")
        self.onnx_dir = os.path.join(model_dir, "onnx")
        self.activate()

    def activate(self):
        self.loading = True
        self.pipe: Union[
            TensorRTStableDiffusionPipeline, TensorRTStableDiffusionImg2ImgPipeline
        ] = TensorRTStableDiffusionPipeline.from_pretrained(
            model_id=self.model.model_id,
            onnx_dir=self.onnx_dir,
            engine_dir=self.engine_dir,
            use_auth_token=config.get("hf_token"),
            device=torch.device("cuda"),
            max_batch_size=1,
        )
        self.loading = False
        self.text_encoder = TensorRTCLIPTextModel(
            self.pipe.engine["clip"], self.pipe.stream
        )

        self.lpw = LongPromptWeightingPipeline(
            self.text_encoder, self.pipe.tokenizer, self.pipe.device
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
            return self.lpw(
                prompt,
                negative_prompt,
                num_images_per_prompt,
                max_embeddings_multiples=1,
            ).to(dtype=torch.float16)

        self.pipe._encode_prompt = _encode_prompt

    def teardown(self):
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
                    factor = 1.0 / 0.18215 * latents
                    sample_inp = cuda.DeviceView(
                        ptr=factor.data_ptr(), shape=factor.shape, dtype=np.float32
                    )
                    images = self.pipe.run_engine("vae", {"latent": sample_inp})[
                        "images"
                    ]
                    images = self.pipe.decode_images(images)
                    images = [
                        *images,
                        *list(chain.from_iterable([x for x, _ in results])),
                    ]

                async def runner():
                    data = DenoiseLatentData(
                        step=step,
                        preview={utils.img2b64(x): opts for x in images}
                        if include
                        else [],
                    )
                    await sio.emit("denoise_latent", data=data.dict())

                utils.fire_and_forget(runner)()

            generator = torch.Generator(device=self.pipe.device).manual_seed(
                manual_seed
            )
            images = self.pipe(
                prompt=opts.prompt,
                negative_prompt=opts.negative_prompt,
                image_height=opts.image_height,
                image_width=opts.image_width,
                guidance_scale=opts.scale,
                num_inference_steps=opts.steps,
                generator=generator,
                callback=callback,
            )
            results.append(
                (
                    images,
                    ImageGenerationOptions.parse_obj(
                        {"seed": manual_seed, **opts.dict()}
                    ),
                )
            )

        all_perf_time = time.perf_counter() - e2e_tic
        result = ImageGenerationResult(images={}, performance=all_perf_time)
        for images, opts in results:
            for img in images:
                result.images[utils.img2b64(img)] = opts
                save_image(img, opts)

        return result
