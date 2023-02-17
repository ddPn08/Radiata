from typing import Optional

from pydantic import BaseModel


class BuildEngineOptions(BaseModel):
    model_id: str
    hf_token: Optional[str] = None
    subfolder: Optional[str] = None
    fp16 = False
    verbose = False
    opt_image_height = 512
    opt_image_width = 512
    min_latent_resolution = 256
    max_latent_resolution = 1024
    max_batch_size = 1
    onnx_opset = 16
    build_static_batch = False
    build_dynamic_shape = True
    build_preview_features = False
    force_engine_build = False
    force_onnx_export = False
    force_onnx_optimize = False
    onnx_minimal_optimization = False
