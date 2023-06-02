import gradio as gr

from lib.diffusers.scheduler import SCHEDULERS
from modules import plugin_loader


def prompt_ui():
    with gr.Column(scale=3):
        prompt_textbox = gr.TextArea(
            "",
            lines=3,
            placeholder="Prompt",
            show_label=False,
            elem_classes=["prompt-textbox"],
        )
        negative_prompt_textbox = gr.TextArea(
            "",
            lines=3,
            placeholder="Negative Prompt",
            show_label=False,
            elem_classes=["negative-prompt-textbox"],
        )
    return prompt_textbox, negative_prompt_textbox


def button_ui():
    return gr.Button(
        "Generate",
        variant="primary",
    )


def common_options_ui():
    with gr.Row():
        sampler_dropdown = gr.Dropdown(
            choices=list(SCHEDULERS.keys()),
            value="euler_a",
            label="Sampler",
        )
        sampling_steps_slider = gr.Slider(
            value=25,
            minimum=1,
            maximum=100,
            step=1,
            label="Sampling Steps",
        )
    with gr.Row():
        batch_size_slider = gr.Slider(
            value=1,
            minimum=1,
            maximum=50,
            step=1,
            label="Batch size",
        )
        batch_count_slider = gr.Slider(
            value=1,
            minimum=1,
            maximum=50,
            step=1,
            label="Batch count",
        )
    with gr.Row():
        cfg_scale_slider = gr.Slider(
            value=7.5,
            minimum=1,
            maximum=20,
            step=0.5,
            label="CFG Scale",
        )
        seed_number = gr.Number(
            label="Seed",
            value=-1,
        )
    with gr.Row():
        width_slider = gr.Slider(
            value=512, minimum=64, maximum=2048, step=64, label="Width"
        )
        height_slider = gr.Slider(
            value=512, minimum=64, maximum=2048, step=64, label="Height"
        )
    return (
        sampler_dropdown,
        sampling_steps_slider,
        batch_size_slider,
        batch_count_slider,
        cfg_scale_slider,
        seed_number,
        width_slider,
        height_slider,
    )


def upscale_options_ui():
    with gr.Row():
        with gr.Accordion("Upscaler", open=False):
            with gr.Row():
                enable_hires = gr.Checkbox(label="Hires.fix")
                enable_multidiff = gr.Checkbox(label="Multi-Diffusion")
            with gr.Accordion("Hires.fix Options", open=False):
                with gr.Row():
                    upscaler_mode = gr.Dropdown(
                        choices=[
                            "bilinear",
                            "bilinear-antialiased",
                            "bicubic",
                            "bicubic-antialiased",
                            "nearest",
                            "nearest-exact",
                        ],
                        value="bilinear",
                        label="Latent upscaler mode",
                    )
                    scale_slider = gr.Slider(
                        value=1.5, minimum=1, maximum=4, step=0.05, label="Upscale by"
                    )
            with gr.Accordion("Multi-Diffusion Options", open=False):
                with gr.Row():
                    views_batch_size = gr.Slider(
                        value=4, minimum=1, maximum=32, step=1, label="tile batch size"
                    )
                with gr.Row():
                    window_size = gr.Slider(
                        value=64,
                        minimum=64,
                        maximum=128,
                        step=8,
                        label="window size (latent)",
                    )
                    stride = gr.Slider(
                        value=16, minimum=8, maximum=64, step=8, label="stride (latent)"
                    )

    return (
        enable_hires,
        enable_multidiff,
        upscaler_mode,
        scale_slider,
        views_batch_size,
        window_size,
        stride,
    )


def img2img_options_ui():
    with gr.Column():
        with gr.Accordion("Img2Img", open=False):
            init_image = gr.Image(label="Init Image", type="pil")
            strength_slider = gr.Slider(
                value=0.5, minimum=0, maximum=1, step=0.01, label="Strength"
            )
    return init_image, strength_slider


def plugin_options_ui():
    plugin_values = {}

    with gr.Column():
        for name, data in plugin_loader.plugin_store.items():
            if "ui" not in data or data["ui"] is None:
                continue
            with gr.Accordion(name, open=False):
                plugin_values[name] = data["ui"]()

    return plugin_values
