import gc
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from queue import Queue
from typing import *

import torch

from api.models.diffusion import ImageGenerationOptions

from . import config, utils
from .images import save_image
from .logger import logger
from .shared import hf_diffusers_cache_dir

ModelMode = Literal["diffusers", "tensorrt"]
logged_trt_warning = False


class DiffusersModel:
    def __init__(self, model_id: str):
        self.model_id: str = model_id
        self.mode: ModelMode = "diffusers"
        self.activated: bool = False
        self.pipe = None

    def available_modes(self):
        modes = ["diffusers"]

        if self.trt_available():
            modes.append("tensorrt")

        return modes

    def get_model_dir(self):
        return os.path.join(config.get("model_dir"), self.model_id.replace("/", os.sep))

    def get_trt_path(self):
        return os.path.join(
            config.get("model_dir"),
            "accelerate",
            "tensorrt",
            self.model_id.replace("/", os.sep),
        )

    def trt_available(self):
        global logged_trt_warning
        trt_path = self.get_trt_path()
        necessary_files = [
            "engine/clip.plan",
            "engine/unet.plan",
            "engine/vae.plan",
            "engine/vae_encoder.plan",
            "onnx/clip.opt.onnx",
            "onnx/unet.opt.onnx",
            "onnx/vae.opt.onnx",
            "onnx/vae_encoder.opt.onnx",
        ]
        for file in necessary_files:
            filepath = os.path.join(trt_path, *file.split("/"))
            if not os.path.exists(filepath):
                return False
        trt_module_status, trt_version_status = utils.tensorrt_is_available()
        if config.get("tensorrt"):
            if not trt_module_status or not trt_version_status:
                if not logged_trt_warning:
                    logger.warning(
                        "TensorRT is available, but torch version is not compatible."
                    )
                    logged_trt_warning = True
                return False
        return True

    def activate(self):
        if self.activated:
            return
        if self.mode == "diffusers":
            from .diffusion.pipelines.diffusers import DiffusersPipeline

            self.pipe = DiffusersPipeline.from_pretrained(
                self.model_id,
                use_auth_token=config.get("hf_token"),
                torch_dtype=torch.float16,
                cache_dir=hf_diffusers_cache_dir(),
            ).to(device=torch.device("cuda"))
            self.pipe.enable_attention_slicing()
            if utils.is_installed("xformers") and config.get("xformers"):
                self.pipe.enable_xformers_memory_efficient_attention()
        elif self.mode == "tensorrt":
            from .diffusion.pipelines.tensorrt import TensorRTStableDiffusionPipeline

            model_dir = self.get_trt_path()
            self.pipe = TensorRTStableDiffusionPipeline.from_pretrained(
                model_id=self.model_id,
                engine_dir=os.path.join(model_dir, "engine"),
                use_auth_token=config.get("hf_token"),
                device=torch.device("cuda"),
                max_batch_size=1,
                hf_cache_dir=hf_diffusers_cache_dir(),
            )
        self.activated = True

    def teardown(self):
        if not self.activated:
            return
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        self.activated = False

    def change_mode(self, mode: ModelMode):
        if mode == self.mode:
            return
        self.teardown()
        self.mode = mode
        self.activate()

    def __call__(
        self, opts: ImageGenerationOptions, init_image: Optional[torch.Tensor] = None
    ):
        if not self.activated:
            raise RuntimeError("Model not activated")

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        self.pipe.scheduler = utils.create_scheduler(opts.scheduler_id, self.model_id)

        queue = Queue()
        done = object()
        total_steps = 0

        results = []

        def callback(*args, **kwargs):
            nonlocal total_steps
            total_steps += 1
            queue.put((total_steps, results))

        def on_done(feature):
            queue.put(done)

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

                while True:
                    item = queue.get()
                    if item is done:
                        break
                    yield item

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
