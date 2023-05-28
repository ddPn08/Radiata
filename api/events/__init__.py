import inspect
from dataclasses import dataclass
from typing import *

handlers: Dict[str, List[Callable]] = {}

T = TypeVar("T", bound="BaseEvent")


class BaseEvent:
    __event_name__: ClassVar[str] = ""

    @classmethod
    def register(cls, handler):
        if cls not in handlers:
            handlers[cls] = []
        handlers[cls].append(handler)

    @classmethod
    def call_event(cls: Type[T], *args, **kwargs) -> T:
        if len(args) == 1 and type(args[0]) == cls:
            event = args[0]
        else:
            event = cls(*args, **kwargs)
        if event is None:
            event = cls()
        if not isinstance(event, BaseEvent):
            raise TypeError(
                "Expected event to be an instance of BaseEvent, got {}".format(
                    type(event)
                )
            )

        if cls not in handlers:
            return event

        for handler in handlers[cls]:
            handler(event)

        return event

    def __call__(self):
        fields = self.__dataclass_fields__
        results = []
        for field in fields:
            results.append(getattr(self, field))
        return results


@dataclass
class CancellableEvent(BaseEvent):
    cancelled = False


@dataclass
class SkippableEvent(BaseEvent):
    skip = False


def event_handler():
    def decorator(func: Callable[[BaseEvent], None]):
        sig = inspect.signature(func)
        args_types = sig.parameters.values()
        if len(args_types) < 1:
            return
        e, *_ = args_types
        e = e.annotation
        if issubclass(e, BaseEvent):
            e.register(func)

    return decorator
