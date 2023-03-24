import importlib
import os
from typing import List

from api.models.plugin import PluginData, PluginMetaData
from modules import config

plugins: List[PluginData] = []


def load_plugins():
    plugin_dir = os.path.join(config.ROOT_DIR, "plugins")
    os.makedirs(plugin_dir, exist_ok=True)
    for dir in os.listdir(plugin_dir):
        fullpath = os.path.join(plugin_dir, dir)
        meta_path = os.path.join(fullpath, "plugin.json")
        if not os.path.exists(meta_path):
            continue
        meta = PluginMetaData.parse_file(meta_path)
        main_module = f"plugins.{dir}.{meta.main}"
        try:
            importlib.import_module(main_module, meta.name)
            data = PluginData(
                meta=meta,
                module=main_module,
                dir=fullpath,
                js=os.path.exists(os.path.join(fullpath, "main.js")),
            )
            plugins.append(data)
        except Exception as e:
            print(f"Failed to load plugin: {meta.name}", e)
