from modules.app import app, sio_app

from . import config, model_manager
from .http.api_router import api
from .http.frontend_router import frontend
from .plugin import plugin_loader

sio_app

app.include_router(api)
app.include_router(frontend)

config.init()
model_manager.set_default_model()
plugin_loader.load_plugins()
