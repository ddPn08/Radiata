from typing import List

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
    info = {
        "prompt":"",
    }
    return FileListResponseModel(status="success", length=len(files), data={d:{"info":info} for d in data})
