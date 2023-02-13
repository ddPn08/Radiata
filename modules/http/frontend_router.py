import os

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from ..shared import ROOT_DIR

frontend = APIRouter(prefix="/app", tags=["application"])


@frontend.get("/{full_path:path}")
def handler(_: Request, full_path: str):
    full_path = full_path.replace("/", os.sep)
    if full_path == "":
        full_path = "index.html"
    return FileResponse(os.path.join(ROOT_DIR, "frontend", "app", full_path))
