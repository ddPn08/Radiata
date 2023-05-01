import gradio as gr

from api.models.diffusion import ImageGenerationOptions
from modules import model_manager
from modules.components import image_generation_options
from modules.ui import Tab


def generate_fn(fn):
    def wrapper(
        self,
        prompt: str,
        negative_prompt: str,
        sampler_name: str,
        sampling_steps: int,
        batch_size: int,
        batch_count: int,
        cfg_scale: float,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        strength: float = 0.5,
        init_image=None,
    ):
        opts = ImageGenerationOptions(
            prompt=prompt,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            batch_count=batch_count,
            scheduler_id=sampler_name,
            steps=sampling_steps,
            scale=cfg_scale,
            image_height=height,
            image_width=width,
            strength=strength,
            seed=seed,
            img2img=init_image is not None,
        )
        yield from fn(self, opts, init_image)

    return wrapper


class Txt2Img(Tab):
    def title(self):
        return "Generate"

    def sort(self):
        return 1

    @generate_fn
    def generate_image(
        self,
        opts: ImageGenerationOptions,
        init_image,
    ):
        if model_manager.sd_model is None:
            yield None, "Please select a model.", gr.Button.update()

        yield [], "Generating...", gr.Button.update(
            value="Generating...", variant="secondary", interactive=False
        )

        count = 0

        for data in model_manager.sd_model(opts, init_image):
            if type(data) == tuple:
                step, preview = data
                progress = step / (opts.batch_count * opts.steps)
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
        for images, opts in image:
            results.extend(images)

        yield results, "Finished", gr.Button.update(
            value="Generate", variant="primary", interactive=True
        )

    def ui(self, outlet):
        (
            generate_button,
            prompts,
            options,
            outputs,
        ) = image_generation_options.ui()

        generate_button.click(
            fn=self.generate_image,
            inputs=[*prompts, *options],
            outputs=[*outputs, generate_button],
        )
