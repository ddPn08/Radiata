from typing import *

import json

import gradio as gr
import glob
import os
from modules.components import image_generation_options

from modules import setting
from modules.ui import Tab

from PIL import Image


class ImagesBrowser(Tab):
    def title(self):
        return "Images Browser"

    def sort(self):
        return 3

    def ui(self, outlet):
        with gr.Column():
            outputs_dir = glob.glob(os.path.join("outputs", "*"))
            with gr.Tabs():
                for dir in outputs_dir:
                    name = dir.split(os.sep)[-1]
                    id = f"{name}-gallery"
                    classes = "info-gallery"
                    with gr.Tab(name):
                        outputs_img = glob.glob(os.path.join(dir, "*"))
                        imgs = [f for f in outputs_img if os.path.isfile(f)]
                        imgs_sorted = sorted(imgs, key=os.path.getmtime)
                        gr.Gallery(
                            value=imgs_sorted, elem_classes=classes, elem_id=id
                        ).style(columns=4)

                        info = gr.HTML()
                        info_format: str = "<span>{0}: {1}</span>"

                        def select_image(index: str) -> List[List[str]]:
                            if index == -1:
                                return info.update("")
                            param: dict = Image.open(imgs_sorted[int(index)]).text
                            parameters = param.pop("parameters")
                            try:
                                param.update(json.loads(parameters))
                            except:
                                param.update({"parameters": parameters})

                            value = ", ".join(
                                [
                                    info_format.format(key, param[key])
                                    for key in param.keys()
                                ]
                            )
                            return info.update(value)

                        gr.Button(visible=False, elem_id=f"{id}-button").click(
                            fn=select_image,
                            _js=f"(y)=>[selectedGalleryButton('{id}')]",
                            inputs=[info],
                            outputs=[info],
                        )
