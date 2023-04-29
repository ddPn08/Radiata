from pydantic import BaseModel


class PluginMetaData(BaseModel):
    main: str
    name: str
    author: str
    version: str
    url: str


class PluginData(BaseModel):
    meta: PluginMetaData
    module: str
    dir: str
