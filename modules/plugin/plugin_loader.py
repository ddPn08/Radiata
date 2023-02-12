import importlib
import os

from modules import shared
from modules.plugin.plugin_meta import PluginMetaData


def load_plugins():
    plugin_dir = os.path.join(shared.ROOT_DIR, "plugins")
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
        except Exception as e:
            print(f"Failed to load plugin: {meta.name}", e)
