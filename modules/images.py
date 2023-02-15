import json
import os
import glob
from datetime import datetime
from typing import Dict

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules import config


def get_category(info: Dict):
    if "img2img" in info:
        return "img2img" if info["img2img"] else "txt2img"


def save_image(img: Image.Image, info: Dict):
    parameters = info
    metadata = PngInfo()
    metadata.add_text("parameters", json.dumps(parameters))
    dir = config.get(f"images/{get_category(info)}/save_dir")
    basename: str = config.get(f"images/{get_category(info)}/save_name")
    filename = (
        basename.replace("{seed}", f'{info["seed"]}')
        .replace(
            "{index}", f"{len(os.listdir(dir)) + 1}" if os.path.exists(dir) else "0"
        )
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

def get_all_image_files(category: str):
    dir = config.get(f"images/{category}/save_dir")
    files = glob.glob(os.path.join(dir, "**/*"), recursive=True)
    files = sorted([f.replace(os.sep, "/") for f in files if os.path.isfile(f)], key=os.path.getmtime)
    return [os.path.relpath(f, dir) for f in files]


