from .base import BaseGenerationEvent


class PreInferenceEvent(BaseGenerationEvent):
    event_name = "pre_inference_event"
