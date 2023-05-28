import importlib
import os
from typing import *

import gradio as gr
import gradio.routes

from modules import model_manager, shared

from .components import header
from .shared import ROOT_DIR


class Tab:
    TABS_DIR = os.path.join(ROOT_DIR, "modules", "tabs")

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def sort(self):
        return 1

    def title(self):
        return ""

    def ui(self, outlet: Callable):
        pass

    def visible(self):
        return True

    def __call__(self):
        children_dir = self.filepath[:-3]
        children = []

        if os.path.isdir(children_dir):
            for file in os.listdir(children_dir):
                if not file.endswith(".py"):
                    continue
                module_name = file[:-3]
                parent = os.path.relpath(Tab.TABS_DIR, Tab.TABS_DIR).replace("/", ".")

                if parent.startswith("."):
                    parent = parent[1:]
                if parent.endswith("."):
                    parent = parent[:-1]

                children.append(
                    importlib.import_module(f"modules.tabs.{parent}.{module_name}")
                )

        children = sorted(children, key=lambda x: x.sort())

        tabs = []

        for child in children:
            attrs = child.__dict__
            tab = [x for x in attrs.values() if issubclass(x, Tab)]
            if len(tab) > 0:
                tabs.append(tab[0])

        def outlet():
            with gr.Tabs():
                for tab in tabs:
                    if not tab.visible():
                        continue
                    with gr.Tab(tab.title()):
                        tab()

        return self.ui(outlet)


def load_tabs() -> List[Tab]:
    tabs = []
    files = os.listdir(os.path.join(ROOT_DIR, "modules", "tabs"))

    for file in files:
        if not file.endswith(".py"):
            continue
        module_name = file[:-3]
        module = importlib.import_module(f"modules.tabs.{module_name}")
        attrs = module.__dict__
        TabClass = [
            x
            for x in attrs.values()
            if type(x) == type and issubclass(x, Tab) and not x == Tab
        ]
        if len(TabClass) > 0:
            tabs.append((file, TabClass[0]))

    tabs = sorted([TabClass(file) for file, TabClass in tabs], key=lambda x: x.sort())
    return tabs


def webpath(fn):
    if fn.startswith(ROOT_DIR):
        web_path = os.path.relpath(fn, ROOT_DIR).replace("\\", "/")
    else:
        web_path = os.path.abspath(fn)

    return f"file={web_path}?{os.path.getmtime(fn)}"


def javascript_html():
    script_js = os.path.join(ROOT_DIR, "script.js")
    head = f'<script type="module" src="{webpath(script_js)}"></script>\n'

    return head


def css_html():
    return f'<link rel="stylesheet" property="stylesheet" href="{webpath(os.path.join(ROOT_DIR, "styles.css"))}">'


def create_head():
    head = ""
    head += css_html()
    head += javascript_html()

    def template_response(*args, **kwargs):
        res = shared.gradio_template_response_original(*args, **kwargs)
        res.body = res.body.replace(b"</head>", f"{head}</head>".encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def create_ui():
    block = gr.Blocks()

    with block:
        if model_manager.mode == "stable-diffusion":
            header.ui()
        with gr.Tabs(elem_id="radiata-root"):
            for tab in load_tabs():
                if not tab.visible():
                    continue
                with gr.Tab(tab.title()):
                    tab()

    create_head()

    return block


if not hasattr(shared, "gradio_template_response_original"):
    shared.gradio_template_response_original = gradio.routes.templates.TemplateResponse
