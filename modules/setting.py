from typing import *

import gradio as gr

from modules import config


class DisplaySetting:
    def __init__(self, key: str, title: str, ui_class, ui_kwargs={}) -> None:
        self.key = key
        self.title = title
        self.ui_class = ui_class
        self.ui_kwargs = ui_kwargs
        self.ui = None

    def make_ui(self):
        self.ui = self.ui_class(
            value=config.get(self.key), label=self.title, **self.ui_kwargs
        )
        return self.ui

    def on_change(self, value):
        config.set(self.key, value)


display_settings: Dict[str, List[DisplaySetting]] = {
    "Common": [
        DisplaySetting(
            "common.output-dir-txt2img",
            "Txt2Img output directory",
            gr.Textbox,
        ),
        DisplaySetting(
            "common.output-name-txt2img",
            "Txt2Img output name",
            gr.Textbox,
        ),
        DisplaySetting(
            "common.output-dir-img2img",
            "Img2Img output directory",
            gr.Textbox,
        ),
        DisplaySetting(
            "common.output-name-img2img",
            "Img2Img output name",
            gr.Textbox,
        ),
    ],
    "Acceleration": [
        DisplaySetting(
            "acceleration.tensorrt.full-acceleration",
            "Full Acceleration",
            gr.Checkbox,
        )
    ],
}
