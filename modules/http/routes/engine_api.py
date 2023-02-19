import time

from fastapi.responses import StreamingResponse

from typing import Optional, Union

from fastapi.responses import StreamingResponse

from api.tensorrt import (
    BuildEngineOptions,
    BuildEngineError,
    BuildEngineProgress,
)

from modules import runners

from ...diffusion.tensorrt.engine import EngineBuilder
from ..api_router import api


build_thread = None


def dummy_builder():
    for i in range(10):
        time.sleep(5)
        print(i)
        yield b"test"


@api.post("/engine/build", response_model=Union[str, BuildEngineError, BuildEngineProgress])
def build_engine(req: BuildEngineOptions):
    global build_thread
    if build_thread is not None and build_thread.is_alive():
        return BuildEngineError(message="building another model")

    if runners.current is not None:
        runners.current.teardown()
    runners.current = None

    builder = EngineBuilder(req)

    def generator():
        try:
            for data in builder.build(generator=True, on_end=lambda: runners.set_default_model()):
                yield data.ndjson()
        except Exception as e:
            yield BuildEngineError(message=str(e)).ndjson()
            raise e

    return StreamingResponse(generator())
