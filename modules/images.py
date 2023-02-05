import json
import os
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules import config


def save_image(img: Image.Image, info: dict):
    parameters = info
    metadata = PngInfo()
    metadata.add_text("parameters", json.dumps(parameters))
    dir = config.get("images/txt2img/save_dir")
    basename: str = config.get("images/txt2img/save_name")
    filename = (
        basename.replace("{seed}", f'{info["seed"]}')
        .replace("{prompt}", info["prompt"][:20].replace(" ", "_"))
        .replace("{date}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    img.save(filepath, metadata=metadata)
    return os.path.basename(filepath)


def get_image(category: str, filename: str):
    dir = config.get(f"images/{category}/save_dir")
    filepath = os.path.join(dir, filename)
    img = Image.open(filepath)
    return img
