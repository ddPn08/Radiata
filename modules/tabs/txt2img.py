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
            seed=seed,
        )
        yield from fn(self, opts)

    return wrapper


class Txt2Img(Tab):
    def title(self):
        return "txt2img"

    def sort(self):
        return 1

    @generate_fn
    def generate_image(
        self,
        opts: ImageGenerationOptions,
    ):
        if model_manager.runner is None:
            yield None, "Please select a model.", gr.Button.update()

        yield [], "Generating...", gr.Button.update(
            value="Generating...", variant="secondary", interactive=False
        )

        count = 0

        for data in model_manager.runner.generate(opts):
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

    @generate_fn
    def if_stage_2(
        self,
        opts: ImageGenerationOptions,
    ):
        if model_manager.mode != "deepfloyd_if":
            yield None, "Current mode is not deepfloyd_if.", gr.Button.update()
        if model_manager.runner is None:
            yield None, "Please select a model.", gr.Button.update()

        yield [], "Generating...", gr.Button.update(
            value="Generating...", variant="secondary", interactive=False
        )

        count = 0

        for data in model_manager.runner.stage_2(opts):
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
            value="Stage 2", variant="secondary", interactive=True
        )

    @generate_fn
    def if_stage_3(
        self,
        opts: ImageGenerationOptions,
    ):
        if model_manager.mode != "deepfloyd_if":
            yield None, "Current mode is not deepfloyd_if.", gr.Button.update()
        if model_manager.runner is None:
            yield None, "Please select a model.", gr.Button.update()

        yield [], "Generating...", gr.Button.update(
            value="Generating...", variant="secondary", interactive=False
        )

        count = 0

        for data in model_manager.runner.stage_3(opts):
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
            value="Stage 3", variant="secondary", interactive=True
        )

    def ui(self, outlet):
        (
            [generate_button, stage_2_button, stage_3_button],
            prompts,
            options,
            outputs,
        ) = image_generation_options.ui()

        generate_button.click(
            fn=self.generate_image,
            inputs=[*prompts, *options],
            outputs=[*outputs, generate_button],
        )

        stage_2_button.click(
            fn=self.if_stage_2,
            inputs=[*prompts, *options],
            outputs=[*outputs, stage_2_button],
        )

        stage_3_button.click(
            fn=self.if_stage_3,
            inputs=[*prompts, *options],
            outputs=[*outputs, stage_3_button],
        )
