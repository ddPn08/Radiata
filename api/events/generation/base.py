from ...generation import ImageGenerationOptions
from .. import CancellableEvent


class BaseGenerationEvent(CancellableEvent):
    def __init__(self, opts: ImageGenerationOptions):
        self.opts = opts
