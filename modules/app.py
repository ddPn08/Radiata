import mimetypes

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute

from . import config


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


@app.get("/")
def redirect():
    return RedirectResponse("/app")


sio = socketio.AsyncServer(async_mode="asgi")
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)
