import gc
import os

import torch

from api.models.tensorrt import BuildEngineOptions, TensorRTEngineData
from lib.tensorrt.utilities import (
    build_engine,
    create_models,
    export_onnx,
    optimize_onnx,
)
from modules import model_manager
from modules.logger import logger
from modules.shared import hf_diffusers_cache_dir


def create_onnx_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


class EngineBuilder:
    def __init__(self, opts: BuildEngineOptions):
        self.model = model_manager.sd_model
        self.device = "cuda"
        self.opts = opts
        self.models = create_models(
            model_id=self.model.model_id,
            device=torch.device("cuda"),
            use_auth_token=opts.hf_token,
            max_batch_size=opts.max_batch_size,
            hf_cache_dir=hf_diffusers_cache_dir(),
        )

    def build(self):
        model_dir = self.model.get_trt_path()
        engine_dir = os.path.join(model_dir, "engine")
        onnx_dir = os.path.join(model_dir, "onnx")
        os.makedirs(engine_dir, exist_ok=True)
        os.makedirs(onnx_dir, exist_ok=True)
        for model_name, model_data in self.models.items():
            onnx_path = create_onnx_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = create_onnx_path(model_name, onnx_dir)
            if not self.opts.force_onnx_export and os.path.exists(onnx_path):
                logger.info(f"Found cached model: {onnx_path}")
            else:
                logger.info(f"Exporting model: {onnx_path}")
                export_onnx(
                    onnx_path=onnx_path,
                    model_data=model_data,
                    opt_image_height=self.opts.opt_image_height,
                    opt_image_width=self.opts.opt_image_width,
                    onnx_opset=self.opts.onnx_opset,
                    hf_cache_dir=hf_diffusers_cache_dir(),
                )
            if not self.opts.force_onnx_optimize and os.path.exists(onnx_opt_path):
                logger.info(f"Found cached model: {onnx_opt_path}")
            else:
                logger.info(f"Generating optimizing model: {onnx_opt_path}")
                optimize_onnx(
                    onnx_path=onnx_path,
                    onnx_opt_path=onnx_opt_path,
                    model_data=model_data,
                )
        for model_name, model_data in self.models.items():
            model_data.min_latent_shape = self.opts.min_latent_resolution // 8
            model_data.max_latent_shape = self.opts.max_latent_resolution // 8
            engine_path = os.path.join(engine_dir, f"{model_name}.plan")
            onnx_opt_path = create_onnx_path(model_name, onnx_dir)
            if not self.opts.force_engine_build and os.path.exists(engine_path):
                logger.info(f"Found cached engine: {engine_path}")
            else:
                build_engine(
                    engine_path=engine_path,
                    onnx_opt_path=onnx_opt_path,
                    model_data=model_data,
                    opt_image_height=self.opts.opt_image_height,
                    opt_image_width=self.opts.opt_image_width,
                    build_static_batch=self.opts.build_static_batch,
                    build_dynamic_shape=self.opts.build_dynamic_shape,
                    build_all_tactics=self.opts.build_all_tactics,
                    build_enable_refit=self.opts.build_enable_refit,
                    build_preview_features=self.opts.build_preview_features,
                )

        torch.cuda.empty_cache()
        gc.collect()

        data = TensorRTEngineData(
            static_batch=self.opts.build_static_batch,
            max_batch_size=self.opts.max_batch_size,
            refit=self.opts.build_enable_refit,
            dynamic_shape=self.opts.build_dynamic_shape,
            preview_features=self.opts.build_preview_features,
            all_tactics=self.opts.build_all_tactics,
            optimize_height=self.opts.opt_image_height,
            optimize_width=self.opts.opt_image_width,
            min_latent_shape=self.opts.min_latent_resolution,
            max_latent_shape=self.opts.max_latent_resolution,
        )

        with open(os.path.join(model_dir, "engine.json"), mode="w") as f:
            f.write(data.json())
