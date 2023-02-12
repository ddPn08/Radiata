import inspect
from typing import Callable


class BaseEvent:
    handlers = []
    event_name = ""

    @classmethod
    def call_event(cls, event):
        if not isinstance(event, BaseEvent):
            raise TypeError(
                "Expected event to be an instance of BaseEvent, got {}".format(
                    type(event)
                )
            )
        for handler in cls.handlers:
            handler(event)
        return event


class CancellableEvent(BaseEvent):
    def __init__(self):
        self.cancelled = False


def event_handler():
    def decorator(func: Callable[[BaseEvent], None]):
        sig = inspect.signature(func)
        args_types = sig.parameters.values()
        if len(args_types) < 1:
            return
        [e, *_] = args_types
        e = e.annotation
        if issubclass(e, BaseEvent):
            e.handlers.append(func)
        return

    return decorator
