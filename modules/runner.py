import glob
import os

from modules import config, shared

from .diffusion.inference import TensorRTDiffusionRunner
from .images import save_image

current: TensorRTDiffusionRunner = None


def set_runner(
    model_id: str,
    tokenizer_id="openai/clip-vit-large-patch14",
):
    global current
    if current is not None:
        current.teardown()
    try:
        current = TensorRTDiffusionRunner(model_id)
        current.activate(tokenizer_id)
        config.set("model", model_id)
    except RuntimeError:
        print(f"Failed to load model: {model_id}")


def get_runners():
    model_dirs = glob.glob(
        os.path.join(shared.cmd_opts.model_dir, "**", "model_index.json"),
        recursive=True,
    )
    return [
        os.path.relpath(os.path.dirname(x), shared.cmd_opts.model_dir)
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
    if os.path.exists(shared.cmd_opts.model_dir):
        models = glob.glob(
            os.path.join(shared.cmd_opts.model_dir, "**", "model_index.json"),
            recursive=True,
        )

        if len(models) < 1:
            return

        previous = config.get("model")
        if previous is not None:
            model = os.path.join(shared.cmd_opts.model_dir, previous)
            if os.path.exists(model):
                set_runner(previous)
                return

        model_dir = os.path.relpath(
            os.path.dirname(models[0]), shared.cmd_opts.model_dir
        )

        set_runner(model_dir)
        config.set("model", model_dir)
