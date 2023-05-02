from typing import *

from pydantic import BaseModel


class BuildEngineOptions(BaseModel):
    hf_token: Optional[str] = None
    subfolder: Optional[str] = None
    max_batch_size = 1
    opt_image_height = 512
    opt_image_width = 512
    min_latent_resolution = 256
    max_latent_resolution = 1024
    build_enable_refit = True
    build_static_batch = False
    build_dynamic_shape = True
    build_all_tactics = False
    build_preview_features = False
    onnx_opset = 17
    force_engine_build = False
    force_onnx_export = False
    force_onnx_optimize = False


class TensorRTEngineData(BaseModel):
    trt_version: str = "8.6.0"
    max_batch_size: int = 1
    refit: bool = False
    static_batch: bool = False
    dynamic_shape: bool = True
    all_tactics: bool = False
    preview_features: bool = False
    optimize_height: int = 512
    optimize_width: int = 512
    min_latent_shape: int = 256
    max_latent_shape: int = 1024
