from typing import List
import json

from modules import images
from modules.api.models.base import BaseResponseModel

from ..api_router import api

class FileListResponseModel(BaseResponseModel):
    length: int
    data: dict[str,dict] = {}


@api.get("/images/browser/{category}/{page}", response_model=FileListResponseModel)
def get_all_image_files(category: str, page: int):
    count = 20
    files = images.get_all_image_files(category)
    data = files[page*count:page*count+count]
    data = {d:{"info":json.loads(images.get_image(category, d).text.get("parameters","{}"))} for d in data}
    return FileListResponseModel(status="success", length=len(files), data=data)
