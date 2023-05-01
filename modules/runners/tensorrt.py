import gc
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import *

import torch

from api.models.diffusion import ImageGenerationOptions
from modules import config
from modules.diffusion.pipelines.tensorrt import TensorRTStableDiffusionPipeline
from modules.images import save_image
from modules.model import DiffusersModel
from modules.shared import hf_diffusers_cache_dir

from .runner import BaseRunner


class TensorRTDiffusionRunner(BaseRunner):
    def __init__(self, model: DiffusersModel) -> None:
        super().__init__(model)

        model_dir = model.get_trt_path()

        self.engine_dir = os.path.join(model_dir, "engine")
        self.onnx_dir = os.path.join(model_dir, "onnx")

        self.pipe: Optional[TensorRTStableDiffusionPipeline] = None

        self.activate()

    def activate(self):
        self.loading = True
        self.pipe: TensorRTStableDiffusionPipeline = (
            TensorRTStableDiffusionPipeline.from_pretrained(
                model_id=self.model.model_id,
                engine_dir=self.engine_dir,
                use_auth_token=config.get("hf_token"),
                device=torch.device("cuda"),
                max_batch_size=1,
                hf_cache_dir=hf_diffusers_cache_dir(),
            )
        )
        self.loading = False

    def teardown(self):
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def generate(
        self, opts: ImageGenerationOptions, init_image: Optional[torch.Tensor] = None
    ):
        self.wait_loading()

        results = []

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        self.pipe.scheduler = self.get_scheduler(opts.scheduler_id)

        callback, on_done, wait = self.yielder()

        for i in range(opts.batch_count):
            manual_seed = opts.seed + i

            generator = torch.Generator(device=self.pipe.device).manual_seed(
                manual_seed
            )

            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe,
                    prompt=[opts.prompt] * opts.batch_size,
                    negative_prompt=[opts.negative_prompt] * opts.batch_size,
                    height=opts.image_height,
                    width=opts.image_width,
                    guidance_scale=opts.scale,
                    num_inference_steps=opts.steps,
                    generator=generator,
                    strength=opts.strength,
                    callback=callback,
                    image=init_image,
                )
                feature.add_done_callback(on_done)

                yield from wait()

                images = feature.result().images

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

        yield results
