import glob
import os
from typing import *

import gradio as gr

from modules import config
from modules.components import gallery
from modules.ui import Tab


class ImagesBrowser(Tab):
    def title(self):
        return "Images Browser"

    def sort(self):
        return 2.5

    def ui(self, outlet):
        outputs_dir = config.get("common.image-browser-dir") or "outputs"
        max_img_len = 30

        def get_image(dir: str) -> List[str]:
            return glob.glob(os.path.join(dir, "*.png"))

        # https://github.com/gradio-app/gradio/issues/3793
        with gr.Tabs():
            dir_name_list = [
                dir.split(os.sep)[-1]
                for dir in glob.glob(os.path.join(outputs_dir, "*"))
                if os.path.isdir(dir)
            ]
            tab_selected = gr.State(
                dir_name_list[0] if len(dir_name_list) > 0 else None
            )
            for name in dir_name_list:
                with gr.Tab(name) as tab:
                    tab_name = gr.State(name)
                    tab.select(lambda x: x, tab_name, tab_selected)

                    with gr.Row():
                        prev_btn = gr.Button("Prev Page")
                        page_box = gr.Number(None, label="Page")
                        page_reload_btn = gr.Button("🔄", elem_classes="tool-button")
                        next_btn = gr.Button("Next Page")

                    gallery_box = gallery.outputs_gallery_info_ui(
                        elem_classes="image_generation_gallery", show_label=False
                    ).style(columns=6)

                    def change_page(page: float, tab: str):
                        imgs = get_image(os.path.join(outputs_dir, tab))
                        img_len = len(imgs)
                        page = int(page)
                        if page < 1 or img_len < 1:
                            page = 1
                        elif page > (img_len - 1) // max_img_len + 1:
                            page = (img_len - 1) // max_img_len + 1

                        g_img = [f for f in imgs if os.path.isfile(f)]
                        g_img = sorted(g_img, key=os.path.getmtime, reverse=True)
                        g_img = g_img[(page - 1) * max_img_len : page * max_img_len]

                        [gallery_box.temp_files.add(f) for f in g_img]

                        return (
                            page_box.update(page),
                            gallery_box.update(g_img),
                        )

                    prev_btn.click(lambda x: page_box.update(x - 1), page_box, page_box)
                    next_btn.click(lambda x: page_box.update(x + 1), page_box, page_box)
                    page_reload_btn.click(
                        lambda x: page_box.update(x + 0.1), page_box, page_box
                    )
                    page_box.change(
                        fn=change_page,
                        inputs=[page_box, tab_selected],
                        outputs=[page_box, gallery_box],
                    )
