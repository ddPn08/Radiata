import gc
import os
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import *

import torch

from api.models.diffusion import ImageGenerationOptions
from lib.diffusers.scheduler import SCHEDULERS, parser_schedulers_config

from . import config, utils
from .images import save_image
from .shared import hf_diffusers_cache_dir, get_device

ModelMode = Literal["diffusers", "tensorrt"]


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
        trt_path = self.get_trt_path()
        necessary_files = [
            "engine/unet.plan",
            "onnx/unet.opt.onnx",
        ]
        for file in necessary_files:
            filepath = os.path.join(trt_path, *file.split("/"))
            if not os.path.exists(filepath):
                return False
        trt_module_status, trt_version_status = utils.tensorrt_is_available()
        if not trt_module_status or not trt_version_status:
            return False
        return config.get("tensorrt")

    def trt_full_acceleration_available(self):
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

        return config.get("acceleration.tensorrt.full-acceleration")

    def activate(self):
        if self.activated:
            return
        device = get_device()
        torch_dtype = torch.float16 if config.get("fp16") else torch.float32

        if self.mode == "diffusers":
            from .diffusion.pipelines.diffusers import DiffusersPipeline

            self.pipe = DiffusersPipeline.from_pretrained(
                self.model_id,
                use_auth_token=config.get("hf_token"),
                torch_dtype=torch_dtype,
                cache_dir=hf_diffusers_cache_dir(),
            ).to(device=device)
            self.pipe.enable_attention_slicing()
            if (
                utils.is_installed("xformers")
                and config.get("xformers")
                and device.type == "cuda"
            ):
                self.pipe.enable_xformers_memory_efficient_attention()
        elif self.mode == "tensorrt":
            from .diffusion.pipelines.tensorrt import TensorRTStableDiffusionPipeline

            model_dir = self.get_trt_path()
            self.pipe = TensorRTStableDiffusionPipeline.from_pretrained(
                model_id=self.model_id,
                engine_dir=os.path.join(model_dir, "engine"),
                use_auth_token=config.get("hf_token"),
                max_batch_size=1,
                device=device,
                hf_cache_dir=hf_diffusers_cache_dir(),
                full_acceleration=self.trt_full_acceleration_available(),
            )
        self.activated = True

    def teardown(self):
        if not self.activated:
            return
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        self.activated = False

    def change_mode(self, mode: ModelMode):
        if mode == self.mode:
            return
        self.teardown()
        self.mode = mode
        self.activate()

    def swap_scheduler(self, scheduler_id: str):
        if not self.activated:
            raise RuntimeError("Model not activated")
        self.pipe.scheduler = SCHEDULERS[scheduler_id].from_config(
            self.pipe.scheduler.config, **parser_schedulers_config(scheduler_id)
        )

    def __call__(self, opts: ImageGenerationOptions, plugin_data: Dict[str, List] = {}):
        if not self.activated:
            raise RuntimeError("Model not activated")

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        self.swap_scheduler(opts.scheduler_id)

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
            manual_seed = int(opts.seed + i)

            generator = torch.Generator(device=self.pipe.device).manual_seed(
                manual_seed
            )

            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe,
                    opts=opts,
                    generator=generator,
                    callback=callback,
                    plugin_data=plugin_data,
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
                    [save_image(img, opts) for img in images],
                    ImageGenerationOptions.parse_obj(
                        {"seed": manual_seed, **opts.dict()}
                    ),
                )
            )

        yield results
