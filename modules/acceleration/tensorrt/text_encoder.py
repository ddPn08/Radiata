import torch
from polygraphy import cuda

from lib.tensorrt.utilities import Engine, device_view


class TensorRTCLIPTextModel:
    def __init__(self, engine: Engine, stream: cuda.Stream) -> None:
        self.engine = engine
        self.stream = stream
        pass

    def __call__(self, text_input_ids: torch.Tensor):
        text_input_ids_inp = device_view(text_input_ids)
        text_embeddings = self.engine.infer(
            {"input_ids": text_input_ids_inp}, self.stream
        )["text_embeddings"]
        return [text_embeddings]
