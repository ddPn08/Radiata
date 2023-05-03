import importlib
import os
from glob import glob

import toml

from api.models.plugin import PluginMetaData

from .shared import ROOT_DIR

plugin_store = {}


def load_plugins():
    plugin_dir = os.path.join(ROOT_DIR, "plugins")
    for filepath in glob(os.path.join(plugin_dir, "*", "plugin.toml")):
        dir = os.path.dirname(filepath)

        with open(filepath, "r") as f:
            meta = PluginMetaData.parse_obj(toml.load(f))

        dir_basename = os.path.basename(dir)
        plugin_store[dir_basename] = {
            "module": importlib.import_module(f"plugins.{dir_basename}.{meta.main}"),
            "meta": meta,
            "ui": None,
        }


def register_plugin_ui(func):
    module = func.__module__.split(".")

    assert len(module) > 1, "Plugin UI must be in plugins directory"
    assert module[0] == "plugins", "Plugin UI must be in plugins directory"

    plugin_name = module[1]

    if plugin_name not in plugin_store:
        raise ValueError("Plugin not found")

    plugin_store[plugin_name]["ui"] = func
