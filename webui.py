import os

if "--tensorrt" in os.environ.get("COMMANDLINE_ARGS", ""):
    import tensorrt as trt

    from lib.tensorrt.utilities import TRT_LOGGER

    print(f"TensorRT version: {trt.__version__}")
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

from modules import config, model_manager, ui


def pre_load():
    config.init()
    model_manager.init()


def webui():
    pre_load()
    app = ui.create_ui()
    app.queue(64)
    app, local_url, share_url = app.launch(
        server_name=config.get("host"),
        server_port=config.get("port"),
        share=config.get("share"),
    )


if __name__ == "__main__":
    webui()
