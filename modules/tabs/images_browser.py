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
        tab_id = "images-browser"
        outputs_dir = "outputs"
        max_img_len = 30

        def get_image(dir: str) -> List[str]:
            return glob.glob(os.path.join(dir, "*.png"))

        with gr.Tabs(elem_id=tab_id):
            dir_name_list = [
                dir.split(os.sep)[-1]
                for dir in glob.glob(os.path.join(outputs_dir, "*"))
            ]
            for name in dir_name_list:
                id = f"{name}-gallery"
                classes = "info-gallery"
                with gr.Tab(name) as tab:
                    with gr.Row():
                        prev_btn = gr.Button("Prev Page")
                        page_box = gr.Number(1, label="Page")
                        next_btn = gr.Button("Next Page")

                    gallery = gr.Gallery(elem_classes=classes, elem_id=id).style(
                        columns=6
                    )
                    info = gr.HTML()
                    info_format: str = "<span>{0}: {1}</span>"
                    info_btn = gr.Button(visible=False, elem_id=f"{id}-button")

                    def change_page(page: float, index: int, tab: str, flag: bool):
                        imgs = get_image(os.path.join(outputs_dir, tab))
                        img_len = len(imgs)
                        page = int(page)
                        value = ""
                        if page < 1 or img_len < 1:
                            page = 1
                        elif page > (img_len - 1) // max_img_len + 1:
                            page = (img_len - 1) // max_img_len + 1

                        if flag:
                            g_img = [f for f in imgs if os.path.isfile(f)]
                            g_img = sorted(g_img, key=os.path.getmtime)
                            g_img = g_img[(page - 1) * max_img_len : page * max_img_len]

                        select_img = (page - 1) * max_img_len + index
                        if index >= 0 and select_img < len(imgs):
                            imgs = imgs[select_img]
                            param: dict = Image.open(imgs).text
                            parameters = param.pop("parameters")
                            try:
                                param.update(json.loads(parameters))
                            except:
                                param.update({"parameters": parameters})

                            value = ", ".join(
                                info_format.format(key, param[key])
                                for key in param.keys()
                            )

                        if flag:
                            return (
                                page_box.update(page),
                                info.update(value),
                                gallery.update(g_img),
                            )
                        else:
                            return (
                                page_box.update(page),
                                info.update(value),
                            )

                    tab.select(lambda x: page_box.update(-1), page_box, page_box)
                    prev_btn.click(lambda x: page_box.update(x - 1), page_box, page_box)
                    next_btn.click(lambda x: page_box.update(x + 1), page_box, page_box)
                    page_box.change(
                        fn=lambda x, y, z: change_page(x, y, z, True),
                        _js=f"(x,y,z)=>[x,selectedGalleryButton('{id}'),selectedTab('{tab_id}')]",
                        inputs=[page_box, info, gallery],
                        outputs=[page_box, info, gallery],
                    )
                    info_btn.click(
                        fn=lambda x, y, z: change_page(x, y, z, False),
                        _js=f"(x,y,z)=>[x,selectedGalleryButton('{id}'),selectedTab('{tab_id}')]",
                        inputs=[page_box, info, gallery],
                        outputs=[page_box, info],
                    )
