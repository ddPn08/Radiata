from modules import config


def run():
    cfg = config.get_config()

    if "models" in cfg:
        cfg["models"] = [x["model_id"] for x in cfg["models"]]
    else:
        cfg["models"] = ["runwayml/stable-diffusion-v1-5"]

    if "mode" in cfg:
        del cfg["mode"]

    config.save_config(cfg)
