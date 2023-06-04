import time
from typing import *

import gradio as gr
from api.models.diffusion import (
    HiresfixOptions,
    MultidiffusionOptions,
    ImageGenerationOptions,
)
from modules import model_manager
from modules.components import gallery, image_generation_options
from modules.ui import Tab


def generate_fn(fn):
    def wrapper(self, data):
        as_list = [x for x in data.values()]
        # common inputs
        (
            prompt,
            negative_prompt,
            sampler_name,
            sampling_steps,
            batch_size,
            batch_count,
            cfg_scale,
            seed,
            width,
            height,
        ) = as_list[0:10]
        # upscaler inputs
        (
            enable_hires,
            enable_multidiff,
            upscaler_mode,
            scale_slider,
            views_batch_size,
            window_size,
            stride,
        ) = as_list[10:17]
        # i2i inputs
        init_image, strength = as_list[17:19]

        plugin_values = dict(list(data.items())[19:])

        hiresfix = HiresfixOptions(
            enable=enable_hires,
            mode=upscaler_mode,
            scale=scale_slider,
        )

        multidiffusion = MultidiffusionOptions(
            enable=enable_multidiff,
            views_batch_size=views_batch_size,
            window_size=window_size,
            stride=stride,
        )

        opts = ImageGenerationOptions(
            prompt=prompt,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            batch_count=batch_count,
            scheduler_id=sampler_name,
            num_inference_steps=sampling_steps,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            strength=strength,
            seed=seed,
            image=init_image,
            hiresfix=hiresfix,
            multidiffusion=multidiffusion,
        )
        yield from fn(self, opts, plugin_values)

    return wrapper


class Generate(Tab):
    plugin_values = {}

    def title(self):
        return "Generate"

    def sort(self):
        return 1

    def visible(self):
        return model_manager.mode == "stable-diffusion"

    @generate_fn
    def generate_image(self, opts, plugin_values):
        if model_manager.sd_model is None:
            yield None, "Please select a model.", gr.Button.update()

        yield [], "Generating...", gr.Button.update(
            value="Generating...", variant="secondary", interactive=False
        )

        plugin_data = {}

        for plugin, values in Generate.plugin_values.items():
            plugin_data[plugin] = []
            for value in values:
                plugin_data[plugin].append(plugin_values[value])

        count = 0

        # pre-calculate inference steps
        if opts.hiresfix.enable:
            inference_steps = opts.num_inference_steps + int(
                opts.num_inference_steps * opts.strength
            )
        else:
            inference_steps = opts.num_inference_steps

        start = time.perf_counter()

        for data in model_manager.sd_model(opts, plugin_data):
            if type(data) == tuple:
                step, preview = data
                progress = step / (opts.batch_count * inference_steps)
                previews = []
                for images, opts in preview:
                    previews.extend(images)

                if len(previews) == count:
                    update = gr.Gallery.update()
                else:
                    update = gr.Gallery.update(value=previews)
                    count = len(previews)
                yield update, f"Progress: {progress * 100:.2f}%, Step: {step}", gr.Button.update(
                    value="Generating...", variant="secondary", interactive=False
                )
            else:
                image = data

        end = time.perf_counter()

        results = []
        for images, opts in image:
            results.extend(images)

        yield results, f"Finished in {end - start:0.4f} seconds", gr.Button.update(
            value="Generate", variant="primary", interactive=True
        )

    def ui(self, outlet):
        with gr.Column():
            with gr.Row():
                prompts = image_generation_options.prompt_ui()
                generate_button = image_generation_options.button_ui()

            with gr.Row():
                with gr.Column(scale=1.25):
                    options = image_generation_options.common_options_ui()

                    options += image_generation_options.upscale_options_ui()
                    options += image_generation_options.img2img_options_ui()

                    plugin_values = image_generation_options.plugin_options_ui()

                outputs = gallery.outputs_gallery_ui()

        Generate.plugin_values = plugin_values

        plugin_values_list = [x for value in plugin_values.values() for x in value]

        generate_button.click(
            fn=self.generate_image,
            inputs={
                *prompts,
                *options,
                *plugin_values_list,
            },
            outputs=[*outputs, generate_button],
        )
