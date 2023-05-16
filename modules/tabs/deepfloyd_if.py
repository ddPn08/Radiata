import gc
from typing import *

import gradio as gr
import torch

from modules import model_manager
from modules.components import image_generation_options
from modules.components import gallery
from modules.diffusion.pipelines.deepfloyd_if import IFDiffusionPipeline
from modules.ui import Tab


class DeepFloydIF(Tab):
    def __init__(self, filepath: str):
        super().__init__(filepath)

        self.pipe = IFDiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0",
            "DeepFloyd/IF-II-L-v1.0",
            "stabilityai/stable-diffusion-x4-upscaler",
        )

    def title(self):
        return "Deepfloyd IF"

    def sort(self):
        return 1

    def visible(self):
        return model_manager.mode == "deepfloyd_if"

    def swap_model(
        self, IF_I_id: str, IF_II_id: str, IF_III_id: str, mode: str = "auto"
    ):
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipe = IFDiffusionPipeline.from_pretrained(
            IF_I_id, IF_II_id, IF_III_id, mode
        )

        return (
            self.pipe.IF_I_id,
            self.pipe.IF_II_id,
            self.pipe.IF_III_id,
            self.pipe.mode,
        )

    def create_generate_fn(self, stage: Literal["I", "II", "III"]):
        def generate_image(
            prompt, negative_prompt, guidance_scale, num_inference_steps
        ):
            yield [], "Generating...", gr.Button.update(
                value="Generating...", variant="secondary", interactive=False
            )

            count = 0

            fn = getattr(self.pipe, f"stage_{stage}")

            for data in fn(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ):
                if type(data) == tuple:
                    step, preview = data
                    progress = step / (1 * num_inference_steps)
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

            results = []
            for images, _ in image:
                results.extend(images)

            yield results, "Finished", gr.Button.update(
                value=f"Stage {stage}", variant="primary", interactive=True
            )

        return generate_image

    def ui(self, outlet):
        with gr.Row():
            IF_I_model_id = gr.Textbox(label="model stage I", value=self.pipe.IF_I_id)
            IF_II_model_id = gr.Textbox(
                label="model stage II", value=self.pipe.IF_II_id
            )
            IF_III_model_if = gr.Textbox(
                label="model stage III", value=self.pipe.IF_III_id
            )
            mode = gr.Radio(
                choices=[
                    "auto",
                    "lowvram",
                    "medvram",
                    "off_load",
                    "sequential_off_load",
                    "normal",
                ],
                label="mode",
                value="auto",
            )
            apply_model_button = gr.Button("💾", elem_classes=["tool-button"])
            apply_model_button.click(
                fn=self.swap_model,
                inputs=[
                    IF_I_model_id,
                    IF_II_model_id,
                    IF_III_model_if,
                    mode,
                ],
                outputs=[IF_I_model_id, IF_II_model_id, IF_III_model_if, mode],
            )

        with gr.Column():
            with gr.Row():
                with gr.Column(scale=3):
                    prompts = image_generation_options.prompt_ui()
                with gr.Row():
                    stage_1_button = gr.Button("Stage I", variant="primary")
                    stage_2_button = gr.Button("Stage II", variant="primary")
                    stage_3_button = gr.Button("Stage III", variant="primary")

            with gr.Row():
                with gr.Column(scale=1.25):
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=0.5,
                            value=7.5,
                            label="CFG Scale",
                        )
                        num_inference_steps = gr.Slider(
                            minimum=1, maximum=100, step=1, value=50, label="Steps"
                        )

                outputs = gallery.outputs_gallery_ui()

        stage_1_button.click(
            fn=self.create_generate_fn("I"),
            inputs=[
                *prompts,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[*outputs, stage_1_button],
        )

        stage_2_button.click(
            fn=self.create_generate_fn("II"),
            inputs=[
                *prompts,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[*outputs, stage_2_button],
        )

        stage_3_button.click(
            fn=self.create_generate_fn("III"),
            inputs=[
                *prompts,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[*outputs, stage_3_button],
        )
