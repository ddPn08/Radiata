from dataclasses import dataclass
from typing import *

from gradio import Blocks

from . import BaseEvent


@dataclass
class PreAppLaunchEvent(BaseEvent):
    pass


@dataclass
class PostAppLaunchEvent(BaseEvent):
    pass


@dataclass
class PreUICreateEvent(BaseEvent):
    pass


@dataclass
class PostUICreateEvent(BaseEvent):
    app: Optional[Blocks] = None
