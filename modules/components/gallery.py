import json
from typing import *

import gradio as gr
import gradio.blocks
import gradio.utils
from PIL import Image


def outputs_gallery_info_ui(elem_classes=[], **kwargs):
    elem_classes = [*elem_classes, "info-gallery"]
    format: str = "<span>{0}: {1}</span>"

    def get_root(block: gradio.blocks.BlockContext) -> gradio.blocks.Blocks:
        if block.parent is None:
            return block
        return get_root(block.parent)

    def is_secure_path(path: str) -> bool:
        return str(path) in set().union(*get_root(gallery).temp_file_sets)

    def change_page(src: str):
        path = gradio.utils.abspath(src)
        if not is_secure_path(path):
            return info_box.update("")

        param: dict = Image.open(path).text
        try:
            parameters = param.get("parameters")
            param.update(json.loads(parameters))
            if param["parameters"] == parameters:
                del param["parameters"]
        except:
            pass
        value = ", ".join(format.format(key, param[key]) for key in param.keys())
        return info_box.update(value)

    gallery = gr.Gallery(elem_classes=elem_classes, **kwargs)
    info_box = gr.HTML()
    selected_box = gr.Textbox(visible=False, elem_classes="image-generation-selected")

    selected_box.change(change_page, selected_box, info_box)
    return gallery


def outputs_gallery_ui():
    with gr.Column():
        output_images = outputs_gallery_info_ui(
            elem_classes="image_generation_gallery", show_label=False
        ).style(columns=4)
        status_textbox = gr.Textbox(interactive=False, show_label=False)

    return output_images, status_textbox
