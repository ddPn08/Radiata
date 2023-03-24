import asyncio
import base64
import importlib
import io
from typing import List

import numpy as np
import torch
from PIL import Image


def img2b64(img: Image.Image, format="png"):
    buf = io.BytesIO()
    img.save(buf, format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64


def b642img(img: Image.Image):
    return Image.open(io.BytesIO(base64.b64decode(img)))


def ndarr2img(images: np.ndarray):
    images = (
        ((images + 1) * 255 / 2)
        .clamp(0, 255)
        .detach()
        .permute(0, 2, 3, 1)
        .round()
        .type(torch.uint8)
        .cpu()
        .numpy()
    )
    result: List[Image.Image] = []
    for i in range(images.shape[0]):
        result.append(Image.fromarray(images[i]))
    return result


def is_installed(package: str):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def fire_and_forget(f):
    def wrapped(*args, **kwargs):
        def runner():
            return asyncio.run(f(*args, **kwargs))

        asyncio.new_event_loop().run_in_executor(None, runner)

    return wrapped
