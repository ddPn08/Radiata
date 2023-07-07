import copy
import gc
import inspect
import os
from dataclasses import dataclass
from typing import *

import numpy as np
import PIL.Image
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    convert_from_ckpt,
)
from diffusers.utils import PIL_INTERPOLATION, numpy_to_pil, randn_tensor
from packaging.version import Version
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from api.diffusion.pipelines.diffusers import DiffusersPipelineModel
from api.events.generation import LoadResourceEvent, UNetDenoisingEvent
from api.models.diffusion import ImageGenerationOptions
from modules.diffusion.pipelines.lpw import LongPromptWeightingPipeline
from modules.diffusion.upscalers.multidiffusion import (
    Multidiffusion,
    MultidiffusionTensorRT,
)
from modules.shared import ROOT_DIR


@dataclass
class PipeSession:
    plugin_data: Dict[str, Any]
    opts: ImageGenerationOptions


class DiffusersPipeline(DiffusersPipelineModel):
    __mode__ = "diffusers"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_id: str,
        use_auth_token: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        variant: str = "fp16",
        subfolder: Optional[str] = None,
    ):
        checkpooint_path = os.path.join(
            ROOT_DIR, "models", "checkpoints", pretrained_model_id
        )

        if os.path.exists(checkpooint_path) and os.path.isfile(checkpooint_path):
            temporary_pipe = (
                convert_from_ckpt.download_from_original_stable_diffusion_ckpt(
                    checkpooint_path,
                    from_safetensors=pretrained_model_id.endswith(".safetensors"),
                    load_safety_checker=False,
                    device=device,
                ).to(torch_dtype=torch_dtype)
            )
        else:
            temporary_pipe = DiffusionPipeline.from_pretrained(
                pretrained_model_id,
                use_auth_token=use_auth_token,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                variant=variant,
                # device_map="auto",
            ).to(device, torch_dtype)

        vae = temporary_pipe.vae
        text_encoder = temporary_pipe.text_encoder
        text_encoder_2 = (
            temporary_pipe.text_encoder_2
            if hasattr(temporary_pipe, "text_encoder_2")
            else None
        )
        tokenizer = temporary_pipe.tokenizer
        tokenizer_2 = (
            temporary_pipe.tokenizer_2
            if hasattr(temporary_pipe, "tokenizer_2")
            else None
        )
        unet = temporary_pipe.unet
        scheduler = temporary_pipe.scheduler

        del temporary_pipe

        # if torch.__version__ >= Version("2"):
        #     try:
        #         unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)
        #     except:
        #         pass

        gc.collect()
        torch.cuda.empty_cache()

        pipe = cls(
            id=pretrained_model_id,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            device=device,
            dtype=torch_dtype,
        )
        return pipe

    def __init__(
        self,
        id: str,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        text_encoder_2: Optional[CLIPTextModel] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.id = id
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.unet = unet
        self.scheduler = scheduler

        self.device = device
        self.dtype = dtype
        self.multidiff = None

        self.stage_1st = None
        self.session = None

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        models = [
            self.text_encoder,
            self.text_encoder_2,
            self.unet,
        ]
        for model in models:
            if hasattr(model, "to"):
                model.to(device, dtype)

        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        self.vae.to(dtype=torch.float32).to(device)

        return self

    def enterers(self):
        return []

    def swap_vae(self, vae: Optional[str] = None):
        if vae is None:
            checkpoint_path = os.path.join(ROOT_DIR, "models", "checkpoints", self.id)
            if os.path.exists(checkpoint_path):
                temporary_pipe = StableDiffusionPipeline.from_ckpt(
                    checkpoint_path,
                    from_safetensors=self.id.endswith(".safetensors"),
                    load_safety_checker=False,
                    device=self.device,
                )
                self.vae = temporary_pipe.vae
                del temporary_pipe
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    self.id, subfolder="vae", device=self.device
                )
            self.vae.to(self.device, self.dtype)
            return
        if vae.endswith(".safetensors"):
            state_dict = load_file(vae, device=self.device)
        else:
            state_dict = torch.load(vae, map_location=self.device)
        state_dict = state_dict["state_dict"]

        new_state_dict = {}

        for key, value in state_dict.items():
            if not key.startswith("first_stage_model."):
                key = "first_stage_model." + key
            new_state_dict[key] = value

        state_dict = convert_from_ckpt.convert_ldm_vae_checkpoint(
            new_state_dict, self.vae.config
        )
        self.vae = AutoencoderKL.from_config(self.vae.config)
        self.vae.load_state_dict(state_dict)
        self.vae.to(self.device, self.dtype)

    def load_resources(
        self,
        opts: ImageGenerationOptions,
    ):
        num_inference_steps = opts.num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        LoadResourceEvent.call_event(self)

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

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        if self.text_encoder_2 is not None and self.tokenizer_2 is not None:
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            # Define tokenizers and text encoders
            tokenizers = (
                [self.tokenizer, self.tokenizer_2]
                if self.tokenizer is not None
                else [self.tokenizer_2]
            )
            text_encoders = (
                [self.text_encoder, self.text_encoder_2]
                if self.text_encoder is not None
                else [self.text_encoder_2]
            )
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                    )
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(self.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    bs_embed * num_images_per_prompt, seq_len, -1
                )

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            if do_classifier_free_guidance:
                negative_prompt = negative_prompt or ""
                uncond_tokens: List[str]
                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                negative_prompt_embeds_list = []
                for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                    max_length = prompt_embeds.shape[1]
                    uncond_input = tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    negative_prompt_embeds = text_encoder(
                        uncond_input.input_ids.to(self.device),
                        output_hidden_states=True,
                    )
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                    if do_classifier_free_guidance:
                        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                        seq_len = negative_prompt_embeds.shape[1]

                        negative_prompt_embeds = negative_prompt_embeds.to(
                            dtype=text_encoder.dtype, device=self.device
                        )

                        negative_prompt_embeds = negative_prompt_embeds.repeat(
                            1, num_images_per_prompt, 1
                        )
                        negative_prompt_embeds = negative_prompt_embeds.view(
                            batch_size * num_images_per_prompt, seq_len, -1
                        )

                        # For classifier free guidance, we need to do two forward passes.
                        # Here we concatenate the unconditional and text embeddings into a single batch
                        # to avoid doing two forward passes

                    negative_prompt_embeds_list.append(negative_prompt_embeds)

                negative_prompt_embeds = torch.concat(
                    negative_prompt_embeds_list, dim=-1
                )

            pooled_prompt_embeds = pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)

            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            )
        else:
            lpw = LongPromptWeightingPipeline(
                self,
                self.text_encoder,
                self.tokenizer,
            )
            prompt_embeds, negative_prompt_embeds = lpw(
                prompt,
                negative_prompt,
                num_images_per_prompt,
                max_embeddings_multiples=1,
            )
            return prompt_embeds, negative_prompt_embeds, None, None

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
    ):
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
            image = image.to(self.device).to(self.vae.dtype)
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = torch.cat(
                [self.vae.config.scaling_factor * init_latents] * batch_size, dim=0
            )
            shape = init_latents.shape
            noise = randn_tensor(
                shape, generator=generator, device=self.device, dtype=dtype
            )
            latents = self.scheduler.add_noise(init_latents, noise, timestep)
            return latents.to(dtype=dtype)

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
        unet_additional_kwargs: Dict[str, Any],
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

                event = UNetDenoisingEvent.call_event(
                    self,
                    latent_model_input,
                    step,
                    timestep,
                    latents,
                    timesteps,
                    do_classifier_free_guidance,
                    prompt_embeds,
                    extra_step_kwargs,
                    callback,
                    callback_steps,
                    cross_attention_kwargs,
                )

                unet_additional_kwargs = {
                    **unet_additional_kwargs,
                    **event.unet_additional_kwargs,
                }

                latents = event.latents

                if not event.skip:
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        **unet_additional_kwargs,
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

        return latents

    def decode_latents(self, latents):
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def decode_images(self, image: np.ndarray):
        return numpy_to_pil(image)

    def create_output(self, latents: torch.Tensor, output_type: str, return_dict: bool):
        latents = latents.float()

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents / self.vae.config.scaling_factor)

            # 9. Convert to PIL
            image = self.decode_images(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents / self.vae.config.scaling_factor)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    @torch.no_grad()
    def __call__(
        self,
        opts: ImageGenerationOptions,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        plugin_data: Optional[Dict[str, Any]] = {},
        original_size: Tuple[int, int] = (1024, 1024),
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = (1024, 1024),
    ):
        opts = copy.deepcopy(opts)  # deepcopy options to prevent changes in input opts
        self.session = PipeSession(
            plugin_data=plugin_data,
            opts=opts,
        )

        # Hires.fix
        if opts.hiresfix.enable:
            opts.hiresfix.enable, self.stage_1st = False, True
            opts.image = self.__call__(
                opts,
                generator,
                eta,
                latents,
                prompt_embeds,
                negative_prompt_embeds,
                "latent",
                return_dict,
                callback,
                callback_steps,
                cross_attention_kwargs,
                plugin_data,
            ).images
            opts.height = int(opts.height * opts.hiresfix.scale)
            opts.width = int(opts.width * opts.hiresfix.scale)

            opts.image = torch.nn.functional.interpolate(
                opts.image,
                (opts.height // 8, opts.width // 8),
                mode=opts.hiresfix.mode.split("-")[0],
                antialias=True if "antialiased" in opts.hiresfix.mode else False,
            )
            opts.image = self.create_output(opts.image, "pil", True).images[0]

        # 1. Define call parameters
        num_images_per_prompt = 1
        prompt = [opts.prompt] * opts.batch_size
        negative_prompt = [opts.negative_prompt] * opts.batch_size

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = opts.guidance_scale > 1.0

        # 2. Prepare pipeline resources
        self.load_resources(opts=opts)

        # 3. Prepare timesteps
        timesteps, opts.num_inference_steps = self.get_timesteps(
            opts.num_inference_steps, opts.strength if opts.image is not None else None
        )
        latent_timestep = timesteps[:1].repeat(opts.batch_size * num_images_per_prompt)

        # 4. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        enterers = self.enterers()
        for enterer in enterers:
            enterer.__enter__()

        # 5. Preprocess image
        if opts.image is not None:
            opts.image = self.preprocess_image(opts.image, opts.height, opts.width)

        unet_additional_kwargs = {}

        do_classifier_free_guidance = opts.guidance_scale > 1.0

        # 6. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

        if (
            pooled_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        ):
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                (opts.height, opts.width),
                dtype=prompt_embeds.dtype,
            )

            if do_classifier_free_guidance:
                add_text_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, add_text_embeds], dim=0
                )
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            unet_additional_kwargs["added_cond_kwargs"] = {
                "text_embeds": add_text_embeds.to(self.device),
                "time_ids": add_time_ids.to(self.device),
            }

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            vae_scale_factor=vae_scale_factor,
            unet_in_channels=self.unet.config.in_channels,
            image=opts.image,
            timestep=latent_timestep,
            batch_size=opts.batch_size,
            height=opts.height,
            width=opts.width,
            dtype=prompt_embeds.dtype,
            generator=generator,
        )

        torch.cuda.synchronize()

        # 7. Denoising loop
        if opts.multidiffusion.enable:
            # multidiff denoise
            self.multidiff = (
                MultidiffusionTensorRT(self)
                if self.__mode__ == "tensorrt"
                else Multidiffusion(self)
            )
            views = self.multidiff.get_views(
                opts.height,
                opts.width,
                opts.multidiffusion.window_size,
                opts.multidiffusion.stride,
            )
            latents = self.multidiff.views_denoise_latent(
                views=views,
                latents=latents,
                timesteps=timesteps,
                num_inference_steps=opts.num_inference_steps,
                views_batch_size=opts.multidiffusion.views_batch_size,
                guidance_scale=opts.guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                extra_step_kwargs=extra_step_kwargs,
                callback=callback,
                callback_steps=callback_steps,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            self.multidiff = None
        else:
            latents = self.denoise_latent(
                latents=latents,
                timesteps=timesteps,
                num_inference_steps=opts.num_inference_steps,
                guidance_scale=opts.guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                extra_step_kwargs=extra_step_kwargs,
                callback=callback,
                callback_steps=callback_steps,
                cross_attention_kwargs=cross_attention_kwargs,
                unet_additional_kwargs=unet_additional_kwargs,
            )

        torch.cuda.synchronize()

        outputs = self.create_output(latents, output_type, return_dict)

        for enterer in enterers:
            enterer.__exit__(None, None, None)

        if self.stage_1st:
            self.stage_1st = None
            return outputs

        self.session = None

        return outputs

    def enable_xformers_memory_efficient_attention(
        self, attention_op: Optional[Callable] = None
    ):
        self.unet.enable_xformers_memory_efficient_attention(attention_op=attention_op)
        self.vae.enable_xformers_memory_efficient_attention(attention_op=attention_op)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        self.unet.set_attention_slice(slice_size=slice_size)
