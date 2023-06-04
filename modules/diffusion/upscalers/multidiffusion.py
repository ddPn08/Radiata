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

from lib.tensorrt.engine import UNet2DConditionModelEngine
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

    def align_unet_inputs(
        self,
        latent_model_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        views_batch_size: int,
        real_batch_size: int,
    ):
        if (
            isinstance(self.unet, UNet2DConditionModelEngine)
            and views_batch_size != real_batch_size
        ):
            # expand latent to tensorrt batch size
            shape = latent_model_input.shape[1:]
            latent_align = torch.zeros(
                views_batch_size * 2, *shape, device=latent_model_input.device
            )
            latent_align[: real_batch_size * 2, :, :, :] += latent_model_input
            # repeat prompt_embeds for batch
            prompt_embeds_align = torch.cat([prompt_embeds] * views_batch_size)
        else:
            prompt_embeds_align = torch.cat([prompt_embeds] * real_batch_size)
            latent_align = latent_model_input
        return latent_align, prompt_embeds_align

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
        views_batch_size: int = 1,
    ):
        # hijack ancestral schedulers
        self.ancestral = self.hijack_ancestral_scheduler()
        # 6. Define panorama grid and initialize views for synthesis.
        views_batch = [
            views[i : i + views_batch_size]
            for i in range(0, len(views), views_batch_size)
        ]
        views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(
            views_batch
        )
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # 7. multidiffusion denoise loop
        with tqdm(total=num_inference_steps) as progress_bar:
            for step, timestep in enumerate(timesteps):
                count.zero_()
                value.zero_()
                noise = torch.randn_like(latents)
                for j, batch_view in enumerate(views_batch):
                    vb_size = len(batch_view)
                    # get the latents corresponding to the current view coordinates
                    latents_for_view = torch.cat(
                        [
                            latents[:, :, h_start:h_end, w_start:w_end]
                            for h_start, h_end, w_start, w_end in batch_view
                        ]
                    )
                    self.scheduler.__dict__.update(views_scheduler_status[j])

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents_for_view.repeat_interleave(2, dim=0)
                        if do_classifier_free_guidance
                        else latents_for_view
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, timestep
                    )

                    # align unet inputs for batch
                    latent_model_input, prompt_embeds_input = self.align_unet_inputs(
                        latent_model_input=latent_model_input,
                        prompt_embeds=prompt_embeds,
                        views_batch_size=views_batch_size,
                        real_batch_size=vb_size,
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds_input,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = (
                            noise_pred[::2],
                            noise_pred[1::2],
                        )
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                        noise_pred = noise_pred[:vb_size]

                    # compute the previous noisy sample x_t -> x_t-1
                    scheduler_output = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=timestep,
                        sample=latents_for_view,
                        **extra_step_kwargs,
                    )
                    latents_denoised_batch = scheduler_output.prev_sample
                    sigma_up = scheduler_output.sigma_up if self.ancestral else None

                    views_scheduler_status[j] = copy.deepcopy(self.scheduler.__dict__)

                    # extract value from batch
                    for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                        latents_denoised_batch.chunk(vb_size), batch_view
                    ):
                        value[
                            :, :, h_start:h_end, w_start:w_end
                        ] += latents_view_denoised
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
