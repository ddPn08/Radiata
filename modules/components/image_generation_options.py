import gradio as gr

from lib.diffusers.scheduler import SCHEDULERS
from modules import plugin_loader


def ui():
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=3):
                prompt_textbox = gr.TextArea(
                    "",
                    lines=3,
                    placeholder="Prompt",
                    show_label=False,
                )
                negative_prompt_textbox = gr.TextArea(
                    "",
                    lines=3,
                    placeholder="Negative Prompt",
                    show_label=False,
                )

            generate_button = gr.Button(
                "Generate",
                variant="primary",
            )

        with gr.Row():
            with gr.Column(scale=1.25):
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

                with gr.Column():
                    with gr.Accordion("Img2Img", open=False):
                        init_image = gr.Image(label="Init Image", type="pil")
                        strength_slider = gr.Slider(
                            value=0.5,
                            minimum=0,
                            maximum=1,
                            step=0.01,
                        )

                plugin_values = {}

                with gr.Column():
                    for name, data in plugin_loader.plugin_store.items():
                        if "ui" not in data and data["ui"] is not None:
                            continue
                        with gr.Accordion(name, open=False):
                            plugin_values[name] = data["ui"]()

            with gr.Column():
                output_images = gr.Gallery(
                    elem_classes="image_generation_gallery"
                ).style(columns=4)
                status_textbox = gr.Textbox(interactive=False, show_label=False)

    prompts = [prompt_textbox, negative_prompt_textbox]
    options = [
        sampler_dropdown,  # sampler name
        sampling_steps_slider,  # num sampling steps
        batch_size_slider,  # batch size
        batch_count_slider,  # batch count
        cfg_scale_slider,  # cfg scale
        width_slider,  # width
        height_slider,  # height
        seed_number,  # seed
        strength_slider,  # strength
        init_image,  # init image
    ]
    outputs = [output_images, status_textbox]

    return generate_button, prompts, options, outputs, plugin_values
