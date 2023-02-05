import importlib
import os

from fastapi import APIRouter

__dirname__ = os.path.dirname(__file__)

api = APIRouter(prefix="/api", tags=["main"])


def initialize():
    for route in os.listdir(os.path.join(__dirname__, "routes")):
        module_name = "modules.api.routes." + route.replace(".py", "")
        importlib.import_module(module_name)


initialize()
