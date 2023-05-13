from typing import *

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
                    with gr.Tab(dir.split(os.sep)[-1]):
                        outputs_img = glob.glob(os.path.join(dir, "*"))
                        imgs = [f for f in outputs_img if os.path.isfile(f)]
                        imgs_sorted = sorted(imgs, key=os.path.getmtime)
                        gr.Gallery(
                            value=[Image.open(img) for img in imgs_sorted]
                        ).style(columns=4)
