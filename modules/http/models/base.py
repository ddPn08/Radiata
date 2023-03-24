from typing import Any, Optional

from pydantic import BaseModel


class BaseResponseModel(BaseModel):
    status: str
    message: Optional[str]
    data: Any
