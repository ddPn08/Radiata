from api.events import event_handler
from api.events.generation import PreInferenceEvent


@event_handler()
def event_listener(e: PreInferenceEvent):
    e.opts.seed = 1
    return
