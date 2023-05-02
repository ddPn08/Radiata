import glob
import json
import os
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from api.models.diffusion import ImageGenerationOptions
from modules import config


def get_category(opts: ImageGenerationOptions):
    return "img2img" if opts.image is not None else "txt2img"


def save_image(img: Image.Image, opts: ImageGenerationOptions):
    metadata = PngInfo()
    metadata.add_text("parameters", opts.json())
    dir = config.get(f"images/{get_category(opts)}/save_dir")
    basename: str = config.get(f"images/{get_category(opts)}/save_name")
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
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    img.save(filepath, pnginfo=metadata)
    return os.path.basename(filepath)


def get_image_filepath(category: str, filename: str):
    dir = config.get(f"images/{category}/save_dir")
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
    dir = config.get(f"images/{category}/save_dir")
    files = glob.glob(os.path.join(dir, "*"))
    files = sorted(
        [f.replace(os.sep, "/") for f in files if os.path.isfile(f)],
        key=os.path.getmtime,
    )
    files.reverse()
    return [os.path.relpath(f, dir) for f in files]
