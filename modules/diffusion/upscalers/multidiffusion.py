import copy
from typing import *

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UNet2DConditionModel,
)
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .samplers import EulerAncestralSampler, KDPM2AncestralSampler


class Multidiffusion:
    def __init__(
        self,
        pipe,
    ):
        self.vae: AutoencoderKL = pipe.text_encoder
        self.text_encoder: CLIPTextModel = pipe.text_encoder
        self.tokenizer: CLIPTokenizer = pipe.tokenizer
        self.unet: UNet2DConditionModel = pipe.unet
        self.scheduler: DDPMScheduler = pipe.scheduler
        self.ancestral = False

    def hijack_ancestral_scheduler(self) -> bool:
        if isinstance(self.scheduler, EulerAncestralDiscreteScheduler):
            config = copy.deepcopy(self.scheduler.__dict__)
            self.scheduler = EulerAncestralSampler.from_config(self.scheduler.config)
            self.scheduler.__dict__.update(config)
            return True
        elif isinstance(self.scheduler, KDPM2AncestralDiscreteScheduler):
            config = copy.deepcopy(self.scheduler.__dict__)
            self.scheduler = KDPM2AncestralSampler.from_config(self.scheduler.config)
            self.scheduler.__dict__.update(config)
            return True
        else:
            return False

    @classmethod
    def get_views(cls, panorama_height, panorama_width, window_size=64, stride=8):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        panorama_height /= 8
        panorama_width /= 8
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return views

    def views_denoise_latent(
        self,
        views: list,
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
        # hijack ancestral schedulers
        self.ancestral = self.hijack_ancestral_scheduler()
        # 6. Define panorama grid and initialize views for synthesis.
        views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(views)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # 7. multidiffusion denoise loop
        with tqdm(total=num_inference_steps) as progress_bar:
            for step, timestep in enumerate(timesteps):
                count.zero_()
                value.zero_()
                noise = torch.randn_like(latents)
                for j, (h_start, h_end, w_start, w_end) in enumerate(views):
                    # get the latents corresponding to the current view coordinates
                    latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]
                    self.scheduler.__dict__.update(views_scheduler_status[j])

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_for_view] * 2)
                        if do_classifier_free_guidance
                        else latents
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
                    scheduler_output = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=timestep,
                        sample=latents_for_view,
                        **extra_step_kwargs,
                    )
                    latents_view_denoised = scheduler_output.prev_sample
                    sigma_up = scheduler_output.sigma_up if self.ancestral else None

                    views_scheduler_status[j] = copy.deepcopy(self.scheduler.__dict__)

                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
                # add noise for ancestral sampler
                latents = (
                    torch.where(count > 0, value / count, value) + noise * sigma_up
                    if sigma_up
                    else torch.where(count > 0, value / count, value)
                )

                # call the callback, if provided
                if step == len(timesteps) - 1 or (
                    (step + 1) > num_warmup_steps
                    and (step + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and step % callback_steps == 0:
                        callback(step, timestep, latents)

        return 1 / 0.18215 * latents
