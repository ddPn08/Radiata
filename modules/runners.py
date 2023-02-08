import glob
import json
import os

from modules import config

from .diffusion.tensorrt.runner import TensorRTDiffusionRunner
from .images import save_image

current: TensorRTDiffusionRunner = None


def set_runner(
    model_dir: str,
    tokenizer_id="openai/clip-vit-large-patch14",
):
    global current
    if current is not None:
        current.teardown()

    meta_path = os.path.join(config.get("model_dir"), model_dir, "model_index.json")

    try:
        with open(meta_path, mode="r") as f:
            meta = json.loads(f.read())
        if "model_id" not in meta:
            meta["model_id"] = os.path.relpath(model_dir).replace(os.sep, "/")
        with open(meta_path, mode="w") as f:
            f.write(json.dumps(meta))
    except Exception as e:
        print(e)

    try:
        current = TensorRTDiffusionRunner(
            os.path.join(config.get("model_dir"), model_dir)
        )
        current.activate(tokenizer_id)
        config.set("model", model_dir)
    except RuntimeError:
        print(f"Failed to load model: {model_dir}")


def get_runners():
    model_dirs = glob.glob(
        os.path.join(config.get("model_dir"), "**", "model_index.json"),
        recursive=True,
    )

    return [
        os.path.relpath(os.path.dirname(x), config.get("model_dir")).replace(
            os.sep, "/"
        )
        for x in model_dirs
    ]


def generate(**kwargs):
    results, all_perf = current.infer(**kwargs)
    all = {}

    for images, info, perf in results:
        for image in images:
            all[save_image(image, info)] = {"info": info, "perf": perf}
    return all, all_perf


def set_default_model():
    if os.path.exists(config.get("model_dir")):
        models = glob.glob(
            os.path.join(config.get("model_dir"), "**", "model_index.json"),
            recursive=True,
        )

        if len(models) < 1:
            return

        previous = config.get("model")
        if previous is not None:
            model = os.path.join(config.get("model_dir"), previous)
            if os.path.exists(model):
                set_runner(previous)
                return

        model_dir = os.path.relpath(os.path.dirname(models[0]), config.get("model_dir"))

        set_runner(model_dir)
        config.set("model", model_dir)
