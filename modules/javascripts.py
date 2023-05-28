import os

from fastapi import FastAPI
from fastapi.responses import FileResponse


def apply_javascript_api(app: FastAPI):
    def get(file_path: str):
        dirname = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        if "." not in basename:
            basename = f"{basename}.js"
        return FileResponse(path=os.path.join("javascripts", dirname, basename))

    app.get("/javascripts/{file_path:path}")(get)
