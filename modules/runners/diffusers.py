import gc
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import *

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from api.models.diffusion import ImageGenerationOptions
from modules import config, utils
from modules.diffusion.lpw import LongPromptWeightingPipeline
from modules.images import save_image
from modules.model import DiffusersModel
from modules.shared import hf_cache_dir

from .runner import BaseRunner


class DiffusersDiffusionRunner(BaseRunner):
    def __init__(self, model: DiffusersModel) -> None:
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
            cache_dir=hf_cache_dir(),
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

    def teardown(self):
        if hasattr(self, "pipe"):
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()

    def generate(self, opts: ImageGenerationOptions):
        self.wait_loading()

        results = []

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        self.pipe.scheduler = self.get_scheduler(opts.scheduler_id)

        for i in range(opts.batch_count):
            manual_seed = opts.seed + i

            generator = torch.Generator(device=self.pipe.device).manual_seed(
                manual_seed
            )

            queue = Queue()
            done = object()

            def callback(
                step: int,
                timestep: torch.Tensor,
                latents: torch.Tensor,
            ):
                queue.put(((opts.steps * i) + step, results))

            def on_done(feature):
                queue.put(done)

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
                    callback=callback,
                )
                feature.add_done_callback(on_done)

                while True:
                    data = queue.get()
                    if data is done:
                        break
                    else:
                        yield data

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
