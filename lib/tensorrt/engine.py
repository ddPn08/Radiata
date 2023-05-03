from typing import *

import torch
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.vae import DecoderOutput
from polygraphy import cuda

from lib.tensorrt.utilities import Engine, device_view


class CLIPTextModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream) -> None:
        self.engine = Engine(filepath)
        self.stream = stream
        self.engine.load()
        self.engine.activate()

    def __call__(self, text_input_ids: torch.Tensor):
        text_embeddings = self.engine.infer(
            {"input_ids": device_view(text_input_ids)}, self.stream
        )["text_embeddings"]
        return [text_embeddings]

    def allocate_buffers(self, shape_dict: Dict, device: torch.device):
        self.engine.allocate_buffers(
            shape_dict=shape_dict,
            device=device,
        )

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream):
        self.engine = Engine(filepath)
        self.stream = stream
        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        sample_inp = device_view(latent_model_input)
        timestep_inp = device_view(timestep)
        embeddings_inp = device_view(encoder_hidden_states)
        noise_pred = self.engine.infer(
            {
                "sample": sample_inp,
                "timestep": timestep_inp,
                "encoder_hidden_states": embeddings_inp,
            },
            self.stream,
        )["latent"]
        return UNet2DConditionOutput(sample=noise_pred)

    def allocate_buffers(self, shape_dict: Dict, device: torch.device):
        self.engine.allocate_buffers(
            shape_dict=shape_dict,
            device=device,
        )

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class AutoencoderKLEngine:
    def __init__(self, encoder_path: str, decoder_path: str, stream: cuda.Stream):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

    def encode(self, images: torch.Tensor):
        return self.encoder.infer({"images": device_view(images)}, self.stream)[
            "latent"
        ]

    def decode(self, latent: torch.Tensor):
        images = self.decoder.infer({"latent": device_view(latent)}, self.stream)[
            "images"
        ]
        return DecoderOutput(sample=images)

    def allocate_buffers(
        self, encoder_shape: Dict, decoder_shape: Dict, device: torch.device
    ):
        self.encoder.allocate_buffers(
            shape_dict=encoder_shape,
            device=device,
        )
        self.decoder.allocate_buffers(
            shape_dict=decoder_shape,
            device=device,
        )

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
