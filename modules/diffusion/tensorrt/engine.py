import gc
import json
import os

import onnx
import torch
from api.tensorrt import BuildEngineOptions

from lib.trt.utilities import Engine
from modules import config

from .models import VAE, UNet


def get_model_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


def build_engine(
    model_name: str,
    engine_dir: str,
    onnx_dir: str,
    model_data,
    opts: BuildEngineOptions,
):
    model_data.min_latent_shape = opts.min_latent_resolution // 8
    model_data.max_latent_shape = opts.max_latent_resolution // 8
    engine = Engine(model_name, engine_dir)
    if opts.force_engine_build or not os.path.exists(engine.engine_path):
        onnx_path = get_model_path(model_name, onnx_dir, opt=False)
        onnx_opt_path = get_model_path(model_name, onnx_dir)
        if not os.path.exists(onnx_opt_path):
            if opts.force_onnx_export or not os.path.exists(onnx_path):
                print(f"Exporting model: {onnx_path}")
                model = model_data.get_model()
                with torch.inference_mode(), torch.autocast("cuda"):
                    inputs = model_data.get_sample_input(
                        1, opts.opt_image_height, opts.opt_image_width
                    )
                    torch.onnx.export(
                        model,
                        inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=opts.onnx_opset,
                        do_constant_folding=True,
                        input_names=model_data.get_input_names(),
                        output_names=model_data.get_output_names(),
                        dynamic_axes=model_data.get_dynamic_axes(),
                    )
                del model
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"Found cached model: {onnx_path}")
            if opts.force_onnx_optimize or not os.path.exists(onnx_opt_path):
                print(f"Generating optimizing model: {onnx_opt_path}")
                onnx_opt_graph = model_data.optimize(
                    onnx.load(onnx_path),
                    minimal_optimization=opts.onnx_minimal_optimization,
                )
                onnx.save(onnx_opt_graph, onnx_opt_path)
        engine.build(
            onnx_opt_path,
            fp16=True,
            input_profile=model_data.get_input_profile(
                1,
                opts.opt_image_height,
                opts.opt_image_width,
                static_batch=opts.build_static_batch,
                static_shape=not opts.build_dynamic_shape,
            ),
            enable_preview=opts.build_preview_features,
        )

    return engine


class EngineBuilder:
    def __init__(self, opts: BuildEngineOptions):
        self.device = "cuda"
        self.opts = opts
        self.models = {
            "unet": UNet(
                opts.model_id,
                hf_token=opts.hf_token,
                subfolder=opts.subfolder,
                fp16=opts.fp16,
                device=self.device,
                verbose=opts.verbose,
                max_batch_size=opts.max_batch_size,
            ),
            "vae": VAE(
                opts.model_id,
                hf_token=opts.hf_token,
                subfolder=opts.subfolder,
                device=self.device,
                verbose=opts.verbose,
                max_batch_size=opts.max_batch_size,
            ),
        }

        self.model_dir = os.path.join(
            config.get("model_dir"),
            os.path.join("__local__", os.path.basename(opts.model_id))
            if os.path.isabs(opts.model_id)
            else opts.model_id,
            opts.subfolder,
        )

    def build(self, generator=False, on_end=lambda: ()):
        engine_dir = os.path.join(self.model_dir, "engine")
        onnx_dir = os.path.join(self.model_dir, "onnx")
        os.makedirs(engine_dir, exist_ok=True)
        os.makedirs(onnx_dir, exist_ok=True)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            if generator:
                yield json.dumps(
                    {
                        "message": f"Building {model_name}...",
                        "progress": i / len(self.models.keys()),
                    }
                )
            engine = build_engine(
                model_name, engine_dir, onnx_dir, model_data, self.opts
            )
            del model_data
            del engine
        meta = {
            "models": {
                "unet": self.models["unet"].model_id,
                "vae": self.models["vae"].model_id,
            },
            "denoising_prec": "fp16" if self.opts.fp16 else "fp32",
            "opt_image_height": self.opts.opt_image_height,
            "opt_image_width": self.opts.opt_image_width,
            "onnx_opset": self.opts.onnx_opset,
            "build_static_batch": self.opts.build_static_batch,
            "build_dynamic_shape": self.opts.build_dynamic_shape,
            "min_latent_resolution": self.opts.min_latent_resolution,
            "max_latent_resolution": self.opts.max_latent_resolution,
            "model_id": self.opts.model_id,
            "subfolder": self.opts.subfolder,
        }
        txt = json.dumps(meta)
        with open(os.path.join(self.model_dir, "model_index.json"), mode="w") as f:
            f.write(txt)
        on_end()
