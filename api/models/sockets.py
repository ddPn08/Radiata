from typing import Dict, Optional, Union
from pydantic import BaseModel


class SocketData(BaseModel):
    namespace: Optional[str]
    event: str
    id: Optional[str]
    data: Union[str, Dict]
