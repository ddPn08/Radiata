from dataclasses import dataclass
from typing import *

from fastapi import FastAPI
from gradio import Blocks

from . import BaseEvent


@dataclass
class PreAppLaunchEvent(BaseEvent):
    pass


@dataclass
class PostAppLaunchEvent(BaseEvent):
    app: FastAPI


@dataclass
class PreUICreateEvent(BaseEvent):
    pass


@dataclass
class PostUICreateEvent(BaseEvent):
    app: Optional[Blocks] = None
