# import glob
# import json
# import os
# from typing import Literal

# from api.models.diffusion import ImageGenerationOptions
# from modules import config, utils
# from modules.images import save_image
# from modules.diffusion.runner import BaseRunner
# from modules.diffusion.tensorrt.runner import TensorRTDiffusionRunner

# current: BaseRunner = None
# mode: Literal["diffusers", "tensorrt"] = "diffusers"


# def meta_filename():
#     if mode == "diffusers":
#         return "model_index.json"
#     elif mode == "tensorrt":
#         return "tensorrt.json"


# def set_runner(
#     model_dir: str,
#     tokenizer_id="openai/clip-vit-large-patch14",
# ):
#     global current
#     if current is not None:
#         current.teardown()

#     meta_path = os.path.join(config.get("model_dir"), model_dir, meta_filename())

#     try:
#         with open(meta_path, mode="r") as f:
#             meta = json.loads(f.read())
#         with open(meta_path, mode="w") as f:
#             f.write(json.dumps(meta))
#     except Exception as e:
#         print(e)

#     try:
#         current = TensorRTDiffusionRunner(
#             os.path.join(config.get("model_dir"), model_dir).replace("/", os.sep)
#         )
#         current.activate(tokenizer_id)
#         config.set("model", model_dir)
#     except RuntimeError:
#         print(f"Failed to load model: {model_dir}")


# def get_runners():
#     model_dirs = glob.glob(
#         os.path.join(config.get("model_dir"), "**", meta_filename()),
#         recursive=True,
#     )

#     return [
#         os.path.relpath(os.path.dirname(x), config.get("model_dir")).replace(
#             os.sep, "/"
#         )
#         for x in model_dirs
#     ]


# def generate(options: ImageGenerationOptions):
#     gen = current.infer(options)
#     result = next(gen)
#     gen.close()

#     for img, info in result.images.items():
#         save_image(utils.b642img(img), info)

#     return result


# def generator(options: ImageGenerationOptions):
#     options.generator = True
#     for data in current.infer(options):
#         yield data
#         if data.type == "result":
#             for img, info in data.images.items():
#                 save_image(utils.b642img(img), info)


# def set_default_model():
#     if os.path.exists(config.get("model_dir")):
#         models = glob.glob(
#             os.path.join(config.get("model_dir"), "**", meta_filename()),
#             recursive=True,
#         )

#         if len(models) < 1:
#             return

#         previous = config.get("model")
#         if previous is not None:
#             model = os.path.join(config.get("model_dir"), previous)
#             if os.path.exists(model):
#                 set_runner(previous)
#                 return

#         model_dir = os.path.relpath(os.path.dirname(models[0]), config.get("model_dir"))

#         set_runner(model_dir)
#         config.set("model", model_dir)
