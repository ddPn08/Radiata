import gc
import json
import os
import random
import time
from typing import Optional, Tuple

import diffusers
import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from polygraphy import cuda
from tqdm import tqdm
from transformers import CLIPTokenizer

from api.events.generation import PreInferenceEvent, PreLatentsCreateEvent
from api.generation import (
    ImageGenerationOptions,
    ImageGenerationResult,
    ImageInformation,
)
from lib.trt.utilities import TRT_LOGGER, Engine
from modules import utils

from ..runner import BaseRunner
from . import clip
from .models import CLIP, VAE, UNet
from .pwp import TensorRTPromptWeightingPipeline


def to_image(images):
    images = (
        ((images + 1) * 255 / 2)
        .clamp(0, 255)
        .detach()
        .permute(0, 2, 3, 1)
        .round()
        .type(torch.uint8)
        .cpu()
        .numpy()
    )
    result = []
    for i in range(images.shape[0]):
        result.append(Image.fromarray(images[i]))
    return result


def get_timesteps(
    scheduler,
    num_inference_steps: int,
    strength: float,
    device: torch.device,
    is_text2img: bool,
) -> Tuple[torch.Tensor, int]:
    if is_text2img:
        return scheduler.timesteps.to(device), num_inference_steps
    else:
        init_timestep = int(num_inference_steps * strength)
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:].to(device)
        return timesteps, num_inference_steps - t_start


schedulers = {
    "ddim": diffusers.DDIMScheduler,
    "deis": diffusers.DEISMultistepScheduler,
    "dpm2": diffusers.KDPM2DiscreteScheduler,
    "dpm2-a": diffusers.KDPM2AncestralDiscreteScheduler,
    "euler_a": diffusers.EulerAncestralDiscreteScheduler,
    "euler": diffusers.EulerDiscreteScheduler,
    "heun": diffusers.DPMSolverMultistepScheduler,
    "dpm++": diffusers.DPMSolverMultistepScheduler,
    "dpm": diffusers.DPMSolverMultistepScheduler,
    "pndm": diffusers.PNDMScheduler,
}


def get_scheduler(scheduler_id: str):
    return schedulers[scheduler_id]


def preprocess_image(image: Image.Image, height: int, width: int):
    width, height = map(lambda x: x - x % 8, (width, height))
    image = image.resize(
        (width, height), resample=diffusers.utils.PIL_INTERPOLATION["lanczos"]
    )
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def prepare_latents(
    vae: VAE,
    vae_scale_factor: int,
    unet_in_channels: int,
    scheduler,
    image: Optional[Image.Image],
    timestep,
    batch_size,
    height,
    width,
    dtype,
    device,
    generator,
    latents=None,
) -> torch.Tensor:
    if image is None:
        shape = (
            batch_size,
            unet_in_channels,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )

        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents
    else:
        init_latent_dist = vae.encode(image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = torch.cat([0.18215 * init_latents] * batch_size, dim=0)
        shape = init_latents.shape
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = scheduler.add_noise(init_latents, noise, timestep)
        return latents


class TensorRTDiffusionRunner(BaseRunner):
    def __init__(self, model_dir: str):
        meta_filepath = os.path.join(model_dir, "model_index.json")
        assert os.path.exists(meta_filepath), "Model meta data not found."
        engine_dir = os.path.join(model_dir, "engine")

        with open(meta_filepath, mode="r") as f:
            txt = f.read()
            self.meta = json.loads(txt)

        self.scheduler = None
        self.scheduler_id = None
        self.model_id = (
            self.meta["model_id"]
            if "model_id" in self.meta
            else "CompVis/stable-diffusion-v1-4"
        )
        self.device = torch.device("cuda")
        self.fp16 = self.meta["denoising_prec"] == "fp16"
        self.engines = {
            "clip": clip.create_clip_engine(),
            "unet": Engine("unet", engine_dir),
            "vae": Engine("vae", engine_dir),
        }
        self.models = {
            "clip": CLIP(clip.model_id, fp16=self.fp16, device=self.device),
            "unet": UNet(
                self.meta["models"]["unet"], subfolder=self.meta["subfolder"], fp16=self.fp16, device=self.device
            ),
            "vae": VAE(self.meta["models"]["vae"], subfolder=self.meta["subfolder"], fp16=self.fp16, device=self.device),
        }
        self.en_vae = self.models["vae"].get_model()

        for model in self.models.values():
            model.min_latent_shape = self.meta["min_latent_resolution"] // 8
            model.max_latent_shape = self.meta["max_latent_resolution"] // 8

    def activate(
        self,
        tokenizer_id="openai/clip-vit-large-patch14",
    ):
        self.stream = cuda.Stream()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_id)
        self.pwp = TensorRTPromptWeightingPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.engines["clip"],
            stream=self.stream,
            device=self.device,
        )

        for engine in self.engines.values():
            engine.activate()
        self.loading = False

    def teardown(self):
        for engine in self.engines.values():
            del engine
        self.stream.free()
        del self.stream
        del self.tokenizer
        del self.pwp.text_encoder
        torch.cuda.empty_cache()
        gc.collect()

    def run_engine(self, model_name, feed_dict):
        engine = self.engines[model_name]
        return engine.infer(feed_dict, self.stream)

    def infer(self, opts: ImageGenerationOptions):
        self.wait_loading()
        opts.img = None if opts.img is None else utils.b642img(opts.img)

        pre_inference_event = PreInferenceEvent(opts)
        PreInferenceEvent.call_event(pre_inference_event)
        # TODO: Implement canceling

        if self.scheduler_id != opts.scheduler_id:
            Scheduler = get_scheduler(opts.scheduler_id)
            try:
                self.scheduler = Scheduler.from_pretrained(
                    self.model_id, subfolder="scheduler"
                )
            except:
                self.scheduler = Scheduler.from_config(
                    {
                        "num_train_timesteps": 1000,
                        "beta_start": 0.00085,
                        "beta_end": 0.012,
                    }
                )

        self.scheduler.set_timesteps(opts.steps, device=self.device)
        timesteps, steps = get_timesteps(
            self.scheduler, opts.steps, opts.strength, self.device, opts.img is None
        )
        latent_timestep = timesteps[:1].repeat(opts.batch_size * opts.batch_count)

        e2e_tic = time.perf_counter()

        results = []

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        for i in range(opts.batch_count):
            for model_name, obj in self.models.items():
                self.engines[model_name].allocate_buffers(
                    shape_dict=obj.get_shape_dict(
                        opts.batch_size, opts.image_height, opts.image_width
                    ),
                    device=self.device,
                )

            manual_seed = opts.seed + i
            generator = torch.Generator(device="cuda").manual_seed(manual_seed)

            with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(
                TRT_LOGGER
            ):
                text_embeddings = self.pwp(
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    guidance_scale=opts.scale,
                    batch_size=opts.batch_size,
                    max_embeddings_multiples=1,
                )
                if self.fp16:
                    text_embeddings = text_embeddings.to(dtype=torch.float16)

                if opts.img is not None:
                    opts.img = preprocess_image(
                        opts.img, opts.image_height, opts.image_width
                    ).to(device=self.device)

                pre_latents_create_event = PreLatentsCreateEvent(opts)
                PreLatentsCreateEvent.call_event(pre_latents_create_event)

                latents = pre_latents_create_event.latents

                if not pre_latents_create_event.skip:
                    latents = prepare_latents(
                        vae=self.en_vae,
                        vae_scale_factor=8,
                        unet_in_channels=4,
                        scheduler=self.scheduler,
                        image=opts.img,
                        timestep=latent_timestep,
                        batch_size=opts.batch_size,
                        height=opts.image_height,
                        width=opts.image_width,
                        dtype=torch.float32,
                        device=self.device,
                        generator=generator,
                    )

                torch.cuda.synchronize()

                for _, timestep in enumerate(tqdm(timesteps)):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, timestep
                    )

                    if timestep.dtype != torch.float32:
                        timestep_float = timestep.float()
                    else:
                        timestep_float = timestep

                    sample_inp = cuda.DeviceView(
                        ptr=latent_model_input.data_ptr(),
                        shape=latent_model_input.shape,
                        dtype=np.float32,
                    )
                    timestep_inp = cuda.DeviceView(
                        ptr=timestep_float.data_ptr(),
                        shape=timestep_float.shape,
                        dtype=np.float32,
                    )
                    embeddings_inp = cuda.DeviceView(
                        ptr=text_embeddings.data_ptr(),
                        shape=text_embeddings.shape,
                        dtype=np.float16 if self.fp16 else np.float32,
                    )

                    noise_pred = self.run_engine(
                        "unet",
                        {
                            "sample": sample_inp,
                            "timestep": timestep_inp,
                            "encoder_hidden_states": embeddings_inp,
                        },
                    )["latent"]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + opts.scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if opts.scheduler_id in [
                        "deis",
                        "dpm2",
                        "heun",
                        "dpm++",
                        "dpm",
                        "pndm",
                    ]:
                        latents = self.scheduler.step(
                            model_output=noise_pred, timestep=timestep, sample=latents
                        ).prev_sample
                    else:
                        latents = self.scheduler.step(
                            model_output=noise_pred,
                            timestep=timestep,
                            sample=latents,
                            generator=generator,
                        ).prev_sample

                latents = 1.0 / 0.18215 * latents
                sample_inp = cuda.DeviceView(
                    ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
                )
                images = self.run_engine("vae", {"latent": sample_inp})["images"]
                torch.cuda.synchronize()

                info = ImageInformation(
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    steps=opts.steps,
                    scale=opts.scale,
                    seed=opts.seed,
                    height=opts.image_height,
                    width=opts.image_width,
                    img2img=opts.img is not None,
                    strength=opts.strength,
                )

                results.append((to_image(images), info))

        e2e_toc = time.perf_counter()
        all_perf_time = e2e_toc - e2e_tic

        result = ImageGenerationResult(images={}, performance=all_perf_time)
        for x in results:
            (images, info) = x
            for img in images:
                result.images[utils.img2b64(img)] = info

        return result
