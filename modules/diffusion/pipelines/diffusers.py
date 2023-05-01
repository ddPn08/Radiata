import gc
import inspect
from typing import *

import numpy as np
import PIL.Image
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import PIL_INTERPOLATION, numpy_to_pil, randn_tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from modules.diffusion.pipelines.lpw import LongPromptWeightingPipeline


class DiffusersPipeline:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_id: str,
        use_auth_token: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        temporary_pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_id,
            use_auth_token=use_auth_token,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        vae = temporary_pipe.vae
        text_encoder = temporary_pipe.text_encoder
        tokenizer = temporary_pipe.tokenizer
        unet = temporary_pipe.unet
        scheduler = temporary_pipe.scheduler

        pipe = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        del temporary_pipe
        torch.cuda.empty_cache()
        gc.collect()
        return pipe

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

        self.lpw = LongPromptWeightingPipeline(self.text_encoder, self.tokenizer)

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        self.vae.to(device, dtype)
        self.text_encoder.to(device, dtype)
        self.unet.to(device, dtype)
        self.tokenizer
        self.scheduler
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def enterers(self):
        return []

    def load_resources(
        self,
        image_height: int,
        image_width: int,
        batch_size: int,
        num_inference_steps: int,
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

    def get_timesteps(self, num_inference_steps: int, strength: Optional[float]):
        if strength is None:
            return self.scheduler.timesteps.to(self.device), num_inference_steps
        else:
            init_timestep = int(num_inference_steps * strength)
            init_timestep = min(init_timestep, num_inference_steps)

            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(self.device)
            return timesteps, num_inference_steps - t_start

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def preprocess_image(self, image: PIL.Image.Image, height: int, width: int):
        width, height = map(lambda x: x - x % 8, (width, height))
        image = image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).contiguous()
        return 2.0 * image - 1.0

    def _encode_prompt(
        self,
        prompt,
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
        )

    def prepare_latents(
        self,
        vae_scale_factor: int,
        unet_in_channels: int,
        image: Optional[torch.Tensor],
        timestep: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: torch.Generator,
        latents: torch.Tensor = None,
    ) -> torch.Tensor:
        if image is None:
            shape = (
                batch_size,
                unet_in_channels,
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

            if latents is None:
                latents = randn_tensor(
                    shape, generator=generator, device=self.device, dtype=dtype
                )
            else:
                if latents.shape != shape:
                    raise ValueError(
                        f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                    )
                latents = latents.to(self.device)

            latents = latents * self.scheduler.init_noise_sigma
            return latents
        else:
            image = image.to(self.device).to(dtype)
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = torch.cat([0.18215 * init_latents] * batch_size, dim=0)
            shape = init_latents.shape
            noise = randn_tensor(
                shape, generator=generator, device=self.device, dtype=dtype
            )
            latents = self.scheduler.add_noise(init_latents, noise, timestep)
            return latents

    def denoise_latent(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
        prompt_embeds: torch.Tensor,
        extra_step_kwargs: Dict[str, Any],
        callback: Optional[Callable],
        callback_steps: int,
        cross_attention_kwargs: Dict[str, Any],
    ):
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with tqdm(total=num_inference_steps) as progress_bar:
            for step, timestep in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    **extra_step_kwargs,
                ).prev_sample

                # call the callback, if provided
                if step == len(timesteps) - 1 or (
                    (step + 1) > num_warmup_steps
                    and (step + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and step % callback_steps == 0:
                        callback(step, timestep, latents)

        return 1 / 0.18215 * latents

    def decode_latents(self, latents):
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def decode_images(self, image: np.ndarray):
        return numpy_to_pil(image)

    def create_output(self, latents: torch.Tensor, output_type: str, return_dict: bool):
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Convert to PIL
            image = self.decode_images(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 1.0,
    ):
        if type(prompt) != list:
            prompt = [prompt]
        if type(negative_prompt) != list:
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(
            negative_prompt
        ), "Prompt and negative prompt must have the same length"

        # 1. Define call parameters
        batch_size = len(prompt)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Prepare pipeline resources
        self.load_resources(
            image_height=height,
            image_width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        )

        # 3. Prepare timesteps
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength if image is not None else None
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 4. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        enterers = self.enterers()
        for enterer in enterers:
            enterer.__enter__()

        # 5. Preprocess image
        if image is not None:
            image = self.preprocess_image(image, height, width)

        # 6. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            vae_scale_factor=8,
            unet_in_channels=4,
            image=image,
            timestep=latent_timestep,
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            generator=generator,
        )

        torch.cuda.synchronize()

        # 7. Denoising loop
        latents = self.denoise_latent(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            extra_step_kwargs=extra_step_kwargs,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        torch.cuda.synchronize()

        outputs = self.create_output(latents, output_type, return_dict)

        for enterer in enterers:
            enterer.__exit__(None, None, None)

        return outputs

    def enable_xformers_memory_efficient_attention(
        self, attention_op: Optional[Callable] = None
    ):
        self.unet.enable_xformers_memory_efficient_attention(attention_op=attention_op)
        self.vae.enable_xformers_memory_efficient_attention(attention_op=attention_op)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        self.unet.set_attention_slice(slice_size=slice_size)
