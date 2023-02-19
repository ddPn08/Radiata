from typing import Dict, Literal, Optional

from ..base import BaseModelStream

class BuildEngineProgress(BaseModelStream):
    type: Literal["result"] = "result"
    message: str
    progress: float

class BuildEngineError(BaseModelStream):
    type: Literal["error"] = "error"
    error: str
    message: str
