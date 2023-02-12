import time

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules import runners

from ...diffusion.tensorrt.engine import EngineBuilder
from ..api_router import api


class BuildRequest(BaseModel):
    model_id: str
    hf_token: str = ""
    fp16: bool = False
    verbose: bool = False
    opt_image_height: int = 512
    opt_image_width: int = 512
    max_batch_size: int = 1
    onnx_opset: int = 16
    build_static_batch: bool = False
    build_dynamic_shape: bool = True
    build_preview_features: bool = False
    force_engine_build: bool = False
    force_onnx_export: bool = False
    force_onnx_optimize: bool = False
    onnx_minimal_optimization: bool = False


build_thread = None


def dummy_builder():
    for i in range(10):
        time.sleep(5)
        print(i)
        yield b"test"


@api.post("/engine/build")
async def build_engine(req: BuildRequest):
    global build_thread
    if build_thread is not None and build_thread.is_alive():
        return {"status": "error", "message": "building another model"}

    if runners.current is not None:
        runners.current.teardown()
    runners.current = None

    builder = EngineBuilder(
        model_id=req.model_id,
        hf_token=req.hf_token,
        fp16=req.fp16,
        verbose=req.verbose,
        opt_image_height=req.opt_image_height,
        opt_image_width=req.opt_image_width,
        max_batch_size=req.max_batch_size,
        onnx_opset=req.onnx_opset,
        build_static_batch=req.build_static_batch,
        build_dynamic_shape=req.build_dynamic_shape,
        build_preview_features=req.build_preview_features,
        force_engine_build=req.force_engine_build,
        force_onnx_export=req.force_onnx_export,
        force_onnx_optimize=req.force_onnx_optimize,
        onnx_minimal_optimization=req.onnx_minimal_optimization,
    )

    return StreamingResponse(
        builder.build(generator=True, on_end=lambda: runners.set_default_model())
    )
