import gc
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import *

import torch
from packaging.version import Version
from diffusers import DiffusionPipeline, IFPipeline
from transformers import T5EncoderModel

from api.models.diffusion import ImageGenerationOptions
from modules import config
from modules.images import save_image
from modules.model import DiffusersModel
from modules.shared import hf_cache_dir

from .runner import BaseRunner


class DeepfloydIFRunner(BaseRunner):
    def __init__(self, model: DiffusersModel, load_after: bool = False) -> None:
        super().__init__(model)
        self.device = torch.device("cuda")
        vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)

        self.pipe_1: Optional[IFPipeline] = None
        self.pipe_2: Optional[IFPipeline] = None
        self.pipe_3: Optional[DiffusionPipeline] = None

        if vram <= 16:
            self.mode = "lowvram"
        elif vram > 16 and vram < 24:
            self.mode = "medvram"
        else:
            self.mode = "highvram"

        if not load_after:
            self.activate()

    def activate(self) -> None:
        self.loading = True

        if self.mode == "highvram":
            self.pipe_1 = self.load_pipeline(1)
            self.pipe_2 = self.load_pipeline(2, text_encoder=None)
            self.pipe_3 = self.load_pipeline(3)

        self.loading = False

    def teardown(self):
        if hasattr(self, "pipe_1"):
            del self.pipe_1
        if hasattr(self, "pipe_2"):
            del self.pipe_2
        if hasattr(self, "pipe_3"):
            del self.pipe_3
        torch.cuda.empty_cache()
        gc.collect()

    def flush(self):
        torch.cuda.empty_cache()
        gc.collect()

    def load_text_encoder(self, **kwargs):
        return T5EncoderModel.from_pretrained(
            self.model.IF_model_id_1,
            subfolder="text_encoder",
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=config.get("hf_token"),
            cache_dir=hf_cache_dir(),
            **kwargs
        )

    def load_pipeline(self, stage: int, **kwargs):
        if stage == 1:
            id = self.model.IF_model_id_1
        elif stage == 2:
            id = self.model.IF_model_id_2
        elif stage == 3:
            id = self.model.IF_model_id_3
        pipe = DiffusionPipeline.from_pretrained(
            id,
            variant="fp16",
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir(),
            use_auth_token=config.get("hf_token"),
            device_map="auto",
            **kwargs
        )
        pipe.safety_checker = None
        # if torch.__version__ < Version("2"):
        #     pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, opts: ImageGenerationOptions):
        self.wait_loading()

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        #
        # create prompt embeds
        #
        if self.mode == "lowvram":
            t5 = self.load_text_encoder()
            pipe = self.load_pipeline(1, text_encoder=t5, unet=None)
            c, uc = pipe.encode_prompt(
                prompt=opts.prompt, negative_prompt=opts.negative_prompt
            )
            del t5
            del pipe
            self.flush()
            self.pipe_1 = self.load_pipeline(1, text_encoder=None)
        else:
            if self.mode == "medvram":
                self.pipe_1 = self.load_pipeline(1)

            c, uc = self.pipe_1.encode_prompt(
                prompt=opts.prompt, negative_prompt=opts.negative_prompt
            )

        images = {}
        results = []

        queue = Queue()
        done = object()
        total_steps = 0

        def callback(
            step: int,
            timestep: torch.Tensor,
            image: torch.Tensor,
        ):
            nonlocal total_steps
            total_steps += 1
            queue.put((total_steps, results))

        def on_done(feature):
            queue.put(done)

        for i in range(opts.batch_count):
            manual_seed = opts.seed + i

            generator = torch.Generator(device=self.device).manual_seed(manual_seed)

            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe_1,
                    prompt_embeds=c,
                    negative_prompt_embeds=uc,
                    # height=opts.image_height,
                    # width=opts.image_width,
                    guidance_scale=opts.scale,
                    # num_inference_steps=opts.steps,
                    generator=generator,
                    callback=callback,
                    output_type="pt",
                )
                feature.add_done_callback(on_done)

                while True:
                    data = queue.get()
                    if data is done:
                        break
                    else:
                        yield data

                result = feature.result().images

                images[manual_seed] = result

        if self.mode == "lowvram" or self.mode == "medvram":
            del self.pipe_1
            self.flush()

            self.pipe_2 = self.load_pipeline(2, text_encoder=None)

        for seed, image in images.items():
            generator = torch.Generator(device=self.device).manual_seed(seed)
            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe_2,
                    image=image,
                    prompt_embeds=c,
                    negative_prompt_embeds=uc,
                    # height=opts.image_height,
                    # width=opts.image_width,
                    guidance_scale=opts.scale,
                    # num_inference_steps=opts.steps,
                    generator=generator,
                    callback=callback,
                    output_type="pt",
                )
                feature.add_done_callback(on_done)

                while True:
                    data = queue.get()
                    if data is done:
                        break
                    else:
                        yield data

                result = feature.result().images

                images[manual_seed] = result

        if self.mode == "lowvram" or self.mode == "medvram":
            del self.pipe_2
            self.flush()

            self.pipe_3 = self.load_pipeline(3)

        for seed, image in images.items():
            generator = torch.Generator(device=self.device).manual_seed(seed)
            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe_3,
                    image=image,
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    height=opts.image_height,
                    width=opts.image_width,
                    # num_inference_steps=opts.steps,
                    generator=generator,
                    noise_level=100,
                    callback=callback,
                )
                feature.add_done_callback(on_done)

                while True:
                    data = queue.get()
                    if data is done:
                        break
                    else:
                        yield data

                result = feature.result().images

                images[manual_seed] = result

        if self.mode == "lowvram" or self.mode == "medvram":
            del self.pipe_3
            self.flush()

        for seed, imgs in images.items():
            option = ImageGenerationOptions.parse_obj({"seed": seed, **opts.dict()})
            results.append((imgs, option))
            for img in imgs:
                save_image(img, option)

        yield results
