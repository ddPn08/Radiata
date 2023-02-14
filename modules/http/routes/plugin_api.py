import os
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from typing import List
from modules.plugin.plugin_loader import plugins, PluginData
from ..api_router import api
from ..models.base import BaseResponseModel


class PluginListResponseModel(BaseResponseModel):
    data: List[PluginData]


@api.get("/plugins/list", response_model=PluginListResponseModel)
def plugin_list():
    return PluginListResponseModel(status="success", data=plugins)


@api.get("/plugins/js/{plugin_name}", response_model=PluginListResponseModel)
def plugin(plugin_name: str):
    plugin = [x for x in plugins if x.meta.name == plugin_name]
    print(plugin, plugin_name, len(plugins))
    if len(plugin) < 1:
        raise HTTPException(status_code=404, detail="Plugin not found")
    else:
        plugin = plugin[0]

    return FileResponse(
        os.path.join(plugin.dir, "main.js"), media_type="application/javascript"
    )
