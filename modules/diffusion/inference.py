import json
import os
import random
import time

import diffusers
import numpy as np
import tensorrt as trt
import torch
import gc
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from transformers import CLIPTokenizer, CLIPTextModel

from lib.trt.utilities import TRT_LOGGER, Engine
from modules import shared

from lib.diffusers.lpw import LongPromptWeightingPipeline
from .clip import create_clip_engine
from .models import CLIP, VAE, UNet


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


def get_scheduler(scheduler_id: str):
    schedulers = {
        "ddim": diffusers.DDIMScheduler,
        "euler_a": diffusers.EulerAncestralDiscreteScheduler,
        "euler": diffusers.EulerDiscreteScheduler,
        "pndm": diffusers.PNDMScheduler,
    }
    return schedulers[scheduler_id]


class TensorRTDiffusionRunner:
    loading = True

    def __init__(
        self,
        model_id: str,
    ):
        meta_filepath = os.path.join(
            shared.cmd_opts.model_dir, model_id, "model_index.json"
        )
        if not os.path.exists(meta_filepath):
            raise RuntimeError("Model meta data not found.")
        engine_dir = os.path.join(shared.cmd_opts.model_dir, model_id, "engine")

        with open(meta_filepath, mode="r") as f:
            txt = f.read()
            self.meta = json.loads(txt)

        self.model_id = model_id
        self.device = "cuda"
        self.fp16 = self.meta["denoising_prec"] == "fp16"
        self.engines = {
            "clip": create_clip_engine(),
            "unet": Engine("unet", engine_dir),
            "vae": Engine("vae", engine_dir),
        }
        self.models = {
            "clip": CLIP("openai/clip-vit-large-patch14"),
            "unet": UNet(self.meta["models"]["unet"], fp16=self.fp16),
            "vae": VAE(self.meta["models"]["vae"]),
        }

    def activate(
        self,
        tokenizer_id="openai/clip-vit-large-patch14",
    ):
        self.stream = cuda.Stream()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_id)
        self.lpw = LongPromptWeightingPipeline(
            tokenizer=self.tokenizer,
            text_encoder=CLIPTextModel.from_pretrained(tokenizer_id).to(self.device),
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
        del self.lpw.text_encoder
        torch.cuda.empty_cache()
        gc.collect()

    def runEngine(self, model_name, feed_dict):
        engine = self.engines[model_name]
        return engine.infer(feed_dict, self.stream)

    def wait_loading(self):
        if not self.loading:
            return
        while self.loading:
            time.sleep(0.5)

    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        batch_size=1,
        batch_count=1,
        scheduler_id: str = "euler_a",
        steps=28,
        scale=28,
        image_height=512,
        image_width=512,
        seed=None,
    ):
        self.wait_loading()
        batch_size = 1

        Scheduler = get_scheduler(scheduler_id)

        scheduler = Scheduler.from_config(self.model_id, subfolder="scheduler")
        scheduler.set_timesteps(steps, device=self.device)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        e2e_tic = time.perf_counter()

        results = []

        if seed is None:
            seed = random.randrange(0, 4294967294, 1)

        for i in range(batch_count):
            # Allocate buffers for TensorRT engine bindings
            for model_name, obj in self.models.items():
                self.engines[model_name].allocate_buffers(
                    shape_dict=obj.get_shape_dict(
                        batch_size, image_height, image_width
                    ),
                    device=self.device,
                )

            # Create profiling events
            events = {}
            for stage in ["clip", "denoise", "vae"]:
                for marker in ["start", "stop"]:
                    events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

            manual_seed = seed + i
            generator = torch.Generator(device="cuda").manual_seed(manual_seed)

            # Run Stable Diffusion pipeline
            with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(
                TRT_LOGGER
            ):
                cudart.cudaEventRecord(events["clip-start"], 0)
                # # Tokenize input
                # text_input_ids = (
                #     self.tokenizer(
                #         prompt,
                #         padding="max_length",
                #         max_length=self.tokenizer.model_max_length,
                #         return_tensors="pt",
                #     )
                #     .input_ids.type(torch.int32)
                #     .to(self.device)
                # )

                # # CLIP text encoder
                # text_input_ids_inp = cuda.DeviceView(
                #     ptr=text_input_ids.data_ptr(),
                #     shape=text_input_ids.shape,
                #     dtype=np.int32,
                # )
                # text_embeddings = self.runEngine(
                #     "clip", {"input_ids": text_input_ids_inp}
                # )["text_embeddings"]

                # # Duplicate text embeddings for each generation per prompt
                # bs_embed, seq_len, _ = text_embeddings.shape
                # text_embeddings = text_embeddings.repeat(1, batch_size, 1)
                # text_embeddings = text_embeddings.view(
                #     bs_embed * batch_size, seq_len, -1
                # )

                # max_length = text_input_ids.shape[-1]
                # uncond_input_ids = (
                #     self.tokenizer(
                #         negative_prompt,
                #         padding="max_length",
                #         max_length=max_length,
                #         truncation=True,
                #         return_tensors="pt",
                #     )
                #     .input_ids.type(torch.int32)
                #     .to(self.device)
                # )
                # uncond_input_ids_inp = cuda.DeviceView(
                #     ptr=uncond_input_ids.data_ptr(),
                #     shape=uncond_input_ids.shape,
                #     dtype=np.int32,
                # )
                # uncond_embeddings = self.runEngine(
                #     "clip", {"input_ids": uncond_input_ids_inp}
                # )["text_embeddings"]

                # # Duplicate unconditional embeddings for each generation per prompt
                # seq_len = uncond_embeddings.shape[1]
                # uncond_embeddings = uncond_embeddings.repeat(1, batch_size, 1)
                # uncond_embeddings = uncond_embeddings.view(batch_size, seq_len, -1)

                # # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
                # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                # if self.fp16:
                #     text_embeddings = text_embeddings.to(dtype=torch.float16)

                text_embeddings = self.lpw(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=scale,
                    num_images_per_prompt=1,
                    max_embeddings_multiples=1,
                )
                if self.fp16:
                    text_embeddings = text_embeddings.to(dtype=torch.float16)

                cudart.cudaEventRecord(events["clip-stop"], 0)

                # latents need to be generated on the target device
                unet_channels = 4  # unet.in_channels
                latents_shape = (
                    batch_size,
                    unet_channels,
                    latent_height,
                    latent_width,
                )
                latents_dtype = torch.float32  # text_embeddings.dtype
                latents = torch.randn(
                    latents_shape,
                    device=self.device,
                    dtype=latents_dtype,
                    generator=generator,
                )

                # Scale the initial noise by the standard deviation required by the scheduler
                latents = latents * scheduler.init_noise_sigma

                torch.cuda.synchronize()

                cudart.cudaEventRecord(events["denoise-start"], 0)
                for step_index, timestep in enumerate(scheduler.timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = scheduler.scale_model_input(
                        latent_model_input, timestep
                    )

                    # predict the noise residual
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
                    noise_pred = self.runEngine(
                        "unet",
                        {
                            "sample": sample_inp,
                            "timestep": timestep_inp,
                            "encoder_hidden_states": embeddings_inp,
                        },
                    )["latent"]

                    # Perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if scheduler_id in ["pndm"]:
                        latents = scheduler.step(
                            model_output=noise_pred, timestep=timestep, sample=latents
                        ).prev_sample
                    else:
                        latents = scheduler.step(
                            model_output=noise_pred,
                            timestep=timestep,
                            sample=latents,
                            generator=generator,
                        ).prev_sample

                latents = 1.0 / 0.18215 * latents
                cudart.cudaEventRecord(events["denoise-stop"], 0)

                cudart.cudaEventRecord(events["vae-start"], 0)
                sample_inp = cuda.DeviceView(
                    ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
                )
                images = self.runEngine("vae", {"latent": sample_inp})["images"]
                cudart.cudaEventRecord(events["vae-stop"], 0)
                torch.cuda.synchronize()

                clip_perf_time = cudart.cudaEventElapsedTime(
                    events["clip-start"], events["clip-stop"]
                )[1]
                denoise_perf_time = cudart.cudaEventElapsedTime(
                    events["denoise-start"], events["denoise-stop"]
                )[1]
                vae_perf_time = cudart.cudaEventElapsedTime(
                    events["vae-start"], events["vae-stop"]
                )[1]

                results.append(
                    (
                        to_image(images),
                        {
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "steps": steps,
                            "scale": scale,
                            "seed": manual_seed,
                            "height": image_height,
                            "width": image_width,
                        },
                        {
                            "clip": clip_perf_time,
                            "denoise": denoise_perf_time,
                            "vae": vae_perf_time,
                        },
                    )
                )

        e2e_toc = time.perf_counter()
        all_perf_time = (e2e_toc - e2e_tic) * 1000

        return results, all_perf_time
