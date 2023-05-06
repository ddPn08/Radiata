import glob
import os
from typing import *

import torch
from safetensors.torch import load_file
from transformers import CLIPTextModel

from api.events import event_handler
from api.events.generation import LoadResourceEvent, PromptTokenizingEvent
from modules.shared import ROOT_DIR

token_replaces = {}
loaded_embeddings = []


@event_handler()
def on_load_resource(e: LoadResourceEvent):
    global token_replaces, loaded_embeddings
    if not isinstance(e.pipe.text_encoder, CLIPTextModel):
        return
    embeddings_dir = os.path.join(ROOT_DIR, "models", "embeddings")
    embeddings = []
    for file in glob.glob(os.path.join(embeddings_dir, "**", "*"), recursive=True):
        safetensors = file.endswith(".safetensors")
        pt = file.endswith(".ckpt") or file.endswith(".pt")
        if safetensors or pt:
            embeddings.append(((pt, safetensors), file))

    if len(embeddings) == len(loaded_embeddings):
        if all(
            [
                embedding in loaded_embeddings
                for embedding in [embedding for _, embedding in embeddings]
            ]
        ):
            return

    token_replaces = {}
    loaded_embeddings = []

    for (pt, safetensors), file in embeddings:
        if safetensors:
            state_dict = load_file(file)
        else:
            state_dict = torch.load(file, map_location="cpu")

        if isinstance(state_dict, torch.Tensor):
            embedding = state_dict
        elif len(state_dict) == 1:
            embedding = next(iter(state_dict.values()))
        elif "string_to_param" in state_dict:
            embedding = state_dict["string_to_param"]["*"]

        token = os.path.splitext(os.path.basename(file))[0]

        embedding = embedding.to(
            dtype=e.pipe.text_encoder.dtype, device=e.pipe.text_encoder.device
        )

        is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

        if is_multi_vector:
            tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
            embeds = [e for e in embedding]  # noqa: C416
        else:
            tokens = [token]
            embeds = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        e.pipe.tokenizer.add_tokens(tokens)
        token_ids = e.pipe.tokenizer.convert_tokens_to_ids(tokens)

        e.pipe.text_encoder.resize_token_embeddings(len(e.pipe.tokenizer))
        for token_id, embedding in zip(token_ids, embeds):
            weight = e.pipe.text_encoder.get_input_embeddings().weight
            if weight.size()[1] != embedding.size()[0]:
                continue
            weight.data[token_id] = embedding

        loaded_embeddings.append(file)
        token_replaces[token_ids[0]] = token_ids


@event_handler()
def on_prompt_tokenizing(e: PromptTokenizingEvent):
    for token in e.text_tokens:
        if token in token_replaces:
            i = e.text_tokens.index(token)
            e.text_tokens[i : i + 1] = token_replaces[token]


def init():
    pass
