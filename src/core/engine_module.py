
from core.events import EventBus
from abc import ABC, abstractmethod
from typing import Optional


class EngineModule(ABC):
    """
    An engine module represent a subsystem of the stepper (like positioner, projector, camera, etc).
    This class provides some basic variable for all the engine modules to use, like the event bus.
    """
    def __init__(self):
        self._event_bus: Optional[EventBus] = None

    @property
    def event_bus(self) -> Optional[EventBus]:
        return self._event_bus

    @event_bus.setter
    def event_bus(self, bus: Optional[EventBus]):
        old_bus = self._event_bus
        if old_bus is not bus:
            if old_bus is not None:
                self._on_detach_event_bus(old_bus)
            self._event_bus = bus
            if bus is not None:
                self._on_attach_event_bus(bus)

    def _on_detach_event_bus(self, bus: EventBus):
        """Override in subclasses to remove event listeners when event bus is detached."""
        pass

    def _on_attach_event_bus(self, bus: EventBus):
        """Override in subclasses to register event listeners when event bus is attached."""
        pass

