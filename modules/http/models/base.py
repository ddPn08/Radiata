from typing import Optional

from pydantic import BaseModel


class BaseResponseModel(BaseModel):
    status: str
    message: Optional[str]
    data: Optional[str]
