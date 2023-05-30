from typing import *

from pydantic import BaseModel


class PluginMetaData(BaseModel):
    main: str
    name: str
    version: str
    author: Optional[str]
    url: Optional[str]
    javascript: Optional[str]
