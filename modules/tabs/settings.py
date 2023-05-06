from typing import *

import gradio as gr

from modules import setting
from modules.ui import Tab


class Settings(Tab):
    def title(self):
        return "Settings"

    def sort(self):
        return 3

    def ui(self, outlet):
        loaded = []
        with gr.Row():
            apply = gr.Button("Apply", variant="primary")
        with gr.Tabs():
            for name, settings in setting.display_settings.items():
                with gr.Tab(name):
                    for x in settings:
                        loaded.append(x.make_ui())

        apply.click(fn=self.apply_settings, inputs={*loaded}, outputs=[apply])

    def apply_settings(self, settings):
        yield gr.Button.update("Applying settings...", variant="secondary")
        for loaded in setting.display_settings.values():
            for x in loaded:
                value = settings[x.ui]
                x.on_change(value)
        yield gr.Button.update("Apply", variant="primary")
