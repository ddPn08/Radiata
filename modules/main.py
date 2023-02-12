import tensorrt
import mimetypes
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute

from . import config, runners
from .http.api_router import api
from .http.frontend_router import frontend
from .plugin import plugin_loader


def custom_generate_unique_id(route: APIRoute):
    return route.name


app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
allowed_hosts = config.get("allow_hosts")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_hosts.split(",") if allowed_hosts else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

app.include_router(api)
app.include_router(frontend)


@app.get("/")
def redirect():
    return RedirectResponse("/app")


def check_model_dir():
    if not os.path.exists(config.get("model_dir")):
        os.makedirs(config.get("model_dir"), exist_ok=True)


config.init()

runners.set_default_model()
plugin_loader.load_plugins()
