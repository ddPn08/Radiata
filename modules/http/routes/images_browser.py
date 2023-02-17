from fastapi.responses import FileResponse

from modules import images
from ..models.base import BaseResponseModel

from ..api_router import api


class FileListResponseModel(BaseResponseModel):
    length: int
    data: dict[str, dict] = {}


@api.get("/images/browser/{category}/{page}", response_model=FileListResponseModel)
def get_all_image_files(category: str, page: int):
    count = 20
    files = images.get_all_image_files(category)
    data = files[page * count : page * count + count]
    data = {
        d: {"info": images.get_image_parameter(images.get_image(category, d))}
        for d in data
    }
    return FileListResponseModel(status="success", length=len(files), data=data)


@api.get("/images/browser/get/{category}/{filename}")
def get_image(category: str, filename: str):
    return FileResponse(images.get_image_filepath(category, filename))
