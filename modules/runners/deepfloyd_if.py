import gc
import random
from concurrent.futures import ThreadPoolExecutor
from typing import *

import torch
from diffusers import DiffusionPipeline, IFPipeline, StableDiffusionUpscalePipeline
from diffusers.utils import pt_to_pil
from transformers import T5EncoderModel

from api.models.diffusion import ImageGenerationOptions
from modules import config
from modules.images import save_image
from modules.model import DiffusersModel
from modules.shared import hf_diffusers_cache_dir, hf_transformers_cache_dir

from .runner import BaseRunner


class DeepfloydIFRunner(BaseRunner):
    def __init__(self, model: DiffusersModel) -> None:
        super().__init__(model)
        self.device = torch.device("cuda")
        vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)

        self.pipe_1: Optional[IFPipeline] = None
        self.pipe_2: Optional[IFPipeline] = None
        self.pipe_3: Optional[StableDiffusionUpscalePipeline] = None

        if vram <= 16:
            self.mode = "lowvram"
        elif vram > 16 and vram < 24:
            self.mode = "medvram"
        else:
            self.mode = "highvram"

        self.previous = {
            "images": [],
            "c": None,
            "uc": None,
        }

    def activate(self) -> None:
        self.loading = True

        if self.mode == "highvram":
            t5 = self.load_text_encoder()
            self.pipe_1 = self.load_pipeline(1, text_encoder=t5)
            self.pipe_2 = self.load_pipeline(2, text_encoder=None)
            self.pipe_3 = self.load_pipeline(3)

        self.loading = False

    def teardown(self):
        self.clear_pipeline()
        torch.cuda.empty_cache()
        gc.collect()

    def flush(self):
        torch.cuda.empty_cache()
        gc.collect()

    def load_text_encoder(self, **kwargs):
        return T5EncoderModel.from_pretrained(
            "DeepFloyd/t5-v1_1-xxl",
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=config.get("hf_token"),
            cache_dir=hf_transformers_cache_dir(),
            **kwargs
        )

    def load_pipeline(self, stage: int, cpu_offload: bool = True, **kwargs):
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
            cache_dir=hf_diffusers_cache_dir(),
            use_auth_token=config.get("hf_token"),
            device_map="auto",
            **kwargs
        )
        pipe.safety_checker = None
        # if torch.__version__ < Version("2"):
        #     pipe.enable_xformers_memory_efficient_attention()
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        return pipe

    def clear_pipeline(self, stage: int = 0):
        if self.mode == "highvram":
            return
        if hasattr(self, "pipe_1") and stage != 1:
            del self.pipe_1
            self.pipe_1 = None
        if hasattr(self, "pipe_2") and stage != 2:
            del self.pipe_2
            self.pipe_2 = None
        if hasattr(self, "pipe_3") and stage != 3:
            del self.pipe_3
            self.pipe_3 = None

        self.flush()

    def generate(self, opts: ImageGenerationOptions):
        print(self.mode)
        self.wait_loading()
        self.clear_pipeline(1)

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        #
        # create prompt embeds
        #
        if self.mode == "lowvram":
            del self.pipe_1
            t5 = self.load_text_encoder()
            self.pipe_1 = self.load_pipeline(
                1, cpu_offload=False, text_encoder=t5, unet=None
            )
            c, uc = self.pipe_1.encode_prompt(
                prompt=opts.prompt, negative_prompt=opts.negative_prompt
            )
            del t5
            del self.pipe_1
            self.flush()
            self.pipe_1 = self.load_pipeline(1, text_encoder=None)
        else:
            if self.mode == "medvram" and self.pipe_1 is None:
                t5 = self.load_text_encoder()
                self.pipe_1 = self.load_pipeline(1, text_encoder=t5)

            c, uc = self.pipe_1.encode_prompt(
                prompt=opts.prompt, negative_prompt=opts.negative_prompt
            )

        # self.pipe_1.scheduler = self.get_scheduler(opts.scheduler_id, self.model.IF_model_id_1)

        images = {}

        callback, on_done, wait = self.yielder()

        for i in range(opts.batch_count):
            manual_seed = opts.seed + i

            generator = torch.Generator(device=self.device).manual_seed(manual_seed)

            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe_1,
                    prompt_embeds=c,
                    negative_prompt_embeds=uc,
                    height=opts.image_height,
                    width=opts.image_width,
                    guidance_scale=opts.scale,
                    num_inference_steps=opts.steps,
                    generator=generator,
                    callback=callback,
                    output_type="pt",
                )
                feature.add_done_callback(on_done)
                yield from wait()

                result = feature.result().images

                images[manual_seed] = result

        results = []

        for seed, imgs in images.items():
            option = ImageGenerationOptions.parse_obj({"seed": seed, **opts.dict()})
            results.append((pt_to_pil(imgs), option))
            # for img in imgs:
            #     save_image(img, option)

        self.previous["images"] = images
        self.previous["c"] = c
        self.previous["uc"] = uc

        yield results

    def stage_2(self, opts: ImageGenerationOptions):
        self.clear_pipeline(2)

        images = self.previous["images"]
        c = self.previous["c"]
        uc = self.previous["uc"]

        if self.mode == "lowvram" or self.mode == "medvram":
            self.pipe_2 = self.load_pipeline(2, text_encoder=None)

        # self.pipe_2.scheduler = self.get_scheduler(opts.scheduler_id, self.model.IF_model_id_2)

        for seed, image in images.items():
            generator = torch.Generator(device=self.device).manual_seed(seed)

            callback, on_done, wait = self.yielder()

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

                yield from wait()

                result = feature.result().images

                images[seed] = result

        results = []

        for seed, imgs in images.items():
            option = ImageGenerationOptions.parse_obj({"seed": seed, **opts.dict()})
            results.append((pt_to_pil(imgs), option))
            # for img in imgs:
            #     save_image(img, option)

        self.previous["images"] = images
        self.previous["c"] = c
        self.previous["uc"] = uc

        yield results

    def stage_3(self, opts: ImageGenerationOptions):
        self.clear_pipeline(3)

        images = self.previous["images"]

        if self.mode == "lowvram" or self.mode == "medvram":
            self.pipe_3 = self.load_pipeline(3)

        # self.pipe_3.scheduler = self.get_scheduler(opts.scheduler_id, self.model.IF_model_id_3)

        for seed, image in images.items():
            generator = torch.Generator(device=self.device).manual_seed(seed)
            callback, on_done, wait = self.yielder()
            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe_3,
                    image=image,
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    # height=opts.image_height,
                    # width=opts.image_width,
                    num_inference_steps=opts.steps,
                    generator=generator,
                    noise_level=100,
                    callback=callback,
                )
                feature.add_done_callback(on_done)

                yield from wait()

                result = feature.result().images

                images[seed] = result

        if self.mode == "lowvram" or self.mode == "medvram":
            del self.pipe_3
            self.pipe_3 = None
            self.flush()

        results = []

        for seed, imgs in images.items():
            option = ImageGenerationOptions.parse_obj({"seed": seed, **opts.dict()})
            results.append((imgs, option))
            for img in imgs:
                save_image(img, option)

        self.previous = {}

        yield results
