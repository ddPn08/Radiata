import glob
import json
import os
import re
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from api.models.diffusion import ImageGenerationOptions
from modules import config


def get_category(opts: ImageGenerationOptions):
    return "img2img" if opts.image is not None else "txt2img"


def replace_invalid_chars(filepath, replace_with="_"):
    invalid_chars = '[\\/:*?"<>|]'

    replace_with = replace_with

    return re.sub(invalid_chars, replace_with, filepath)


def save_image(img: Image.Image, opts: ImageGenerationOptions):
    metadata = PngInfo()
    metadata.add_text("parameters", opts.json())
    dir = config.get(f"common.output-dir-{get_category(opts)}")
    basename: str = config.get(f"common.output-name-{get_category(opts)}")
    filename = (
        basename.format(
            seed=opts.seed,
            index=len(os.listdir(dir)) + 1 if os.path.exists(dir) else 0,
            prompt=opts.prompt[:20].replace(" ", "_"),
            date=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        .replace("\n", "_")
        .replace("\r", "_")
        .replace("\t", "_")
    )
    filename = replace_invalid_chars(filename)
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    img.save(filepath, pnginfo=metadata)
    return os.path.basename(filepath)


def get_image_filepath(category: str, filename: str):
    dir = config.get(f"common.output-dir-{category}")
    return os.path.join(dir, filename)


def get_image(category: str, filename: str):
    return Image.open(get_image_filepath(category, filename))


def get_image_parameter(img: Image.Image):
    text = img.text
    parameters = text.pop("parameters", None)
    try:
        text.update(json.loads(parameters))
    except:
        text.update({"parameters": parameters})
    return text


def get_all_image_files(category: str):
    dir = config.get(f"common.output-dir-{category}")
    files = glob.glob(os.path.join(dir, "*"))
    files = sorted(
        [f.replace(os.sep, "/") for f in files if os.path.isfile(f)],
        key=os.path.getmtime,
    )
    files.reverse()
    return [os.path.relpath(f, dir) for f in files]
