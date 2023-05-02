import os
import time

if "--tensorrt" in os.environ.get("COMMANDLINE_ARGS", ""):
    import tensorrt as trt

    from lib.tensorrt.utilities import TRT_LOGGER

    print(f"TensorRT version: {trt.__version__}")
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

from api.events.common import (
    PostAppLaunchEvent,
    PostUICreateEvent,
    PreAppLaunchEvent,
    PreUICreateEvent,
)
from modules import config, model_manager, plugin_loader, ui


def pre_load():
    config.init()
    plugin_loader.load_plugins()
    PreAppLaunchEvent.call_event()
    model_manager.init()


def post_load():
    PostAppLaunchEvent.call_event()


def wait_on_server():
    while 1:
        time.sleep(0.5)


def webui():
    pre_load()
    PreUICreateEvent.call_event()
    app = ui.create_ui()
    PostUICreateEvent.call_event(PostUICreateEvent(app=app))
    app.queue(64)
    app, local_url, share_url = app.launch(
        server_name=config.get("host"),
        server_port=config.get("port"),
        share=config.get("share"),
        prevent_thread_lock=True,
    )
    post_load()
    wait_on_server()


if __name__ == "__main__":
    webui()
