from typing import *

import gradio as gr

from modules import config, model_manager
from modules.ui import Tab
from modules.utils import tensorrt_is_available


class Txt2Img(Tab):
    def title(self):
        return "TensorRT"

    def sort(self):
        return 2

    def visible(self):
        module, version = tensorrt_is_available()
        return module and version and config.get("tensorrt")

    def ui(self, outlet):
        with gr.Column():
            with gr.Column():
                gr.Markdown("# Build tensorrt engine")
                with gr.Row():
                    max_batch_size_number = gr.Number(
                        label="Max batch size",
                        value=1,
                    )
                    opt_image_height_slider = gr.Slider(
                        label="Image height",
                        minimum=1,
                        maximum=2048,
                        step=64,
                        value=512,
                    )
                    opt_image_width_slider = gr.Slider(
                        label="Image width", minimum=1, maximum=2048, step=64, value=512
                    )
                with gr.Row():
                    min_latent_resolution_slider = gr.Slider(
                        label="Min latent resolution",
                        minimum=1,
                        maximum=2048,
                        step=64,
                        value=256,
                    )
                    max_latent_resolution_slider = gr.Slider(
                        label="Max latent resolution",
                        minimum=1,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )
                with gr.Row():
                    build_enable_refit_checkbox = gr.Checkbox(
                        label="Enable refit",
                        value=False,
                    )
                    build_static_batch_checkbox = gr.Checkbox(
                        label="Static batch",
                        value=False,
                    )
                    build_dynamic_shape_checkbox = gr.Checkbox(
                        label="Dynamic shape",
                        value=True,
                    )
                    build_all_tactics_checkbox = gr.Checkbox(
                        label="All tactics",
                        value=False,
                    )
                    build_preview_features_checkbox = gr.Checkbox(
                        label="Preview features",
                        value=True,
                    )
                with gr.Row():
                    onnx_opset_slider = gr.Slider(
                        label="ONNX opset",
                        minimum=7,
                        maximum=18,
                        step=1,
                        value=17,
                    )
                with gr.Row():
                    force_engine_build_checkbox = gr.Checkbox(
                        label="Force engine build",
                        value=False,
                    )
                    force_onnx_export_checkbox = gr.Checkbox(
                        label="Force ONNX export",
                        value=False,
                    )
                    force_onnx_optimize_checkbox = gr.Checkbox(
                        label="Force ONNX optimize",
                        value=False,
                    )
                status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                )
                build_button = gr.Button(
                    label="Build",
                    variant="primary",
                )
                build_button.click(
                    fn=self.build_engine,
                    inputs=[
                        max_batch_size_number,
                        opt_image_height_slider,
                        opt_image_width_slider,
                        min_latent_resolution_slider,
                        max_latent_resolution_slider,
                        build_enable_refit_checkbox,
                        build_static_batch_checkbox,
                        build_dynamic_shape_checkbox,
                        build_all_tactics_checkbox,
                        build_preview_features_checkbox,
                        onnx_opset_slider,
                        force_engine_build_checkbox,
                        force_onnx_export_checkbox,
                        force_onnx_optimize_checkbox,
                    ],
                    outputs=[status],
                )

    def build_engine(
        self,
        max_batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        min_latent_resolution: int,
        max_latent_resolution: int,
        build_enable_refit: bool,
        build_static_batch: bool,
        build_dynamic_shape: bool,
        build_all_tactics: bool,
        build_preview_features: bool,
        onnx_opset: int,
        force_engine_build: bool,
        force_onnx_export: bool,
        force_onnx_optimize: bool,
    ):
        from api.models.tensorrt import BuildEngineOptions
        from modules.acceleration.tensorrt.engine import EngineBuilder

        yield "Building engine..."
        model_manager.sd_model.teardown()
        opts = BuildEngineOptions(
            max_batch_size=max_batch_size,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            min_latent_resolution=min_latent_resolution,
            max_latent_resolution=max_latent_resolution,
            build_enable_refit=build_enable_refit,
            build_static_batch=build_static_batch,
            build_dynamic_shape=build_dynamic_shape,
            build_all_tactics=build_all_tactics,
            build_preview_features=build_preview_features,
            onnx_opset=onnx_opset,
            force_engine_build=force_engine_build,
            force_onnx_export=force_onnx_export,
            force_onnx_optimize=force_onnx_optimize,
        )
        builder = EngineBuilder(opts)
        builder.build()
        model_manager.sd_model.activate()
        yield "Engine built"
