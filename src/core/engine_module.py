
from core.events import EventBus
from abc import ABC, abstractmethod
from typing import Optional


class EngineModule(ABC):
    """
    An engine module represent a subsystem of the stepper (like positioner, projector, camera, etc).
    This class provides some basic variable for all the engine modules to use, like the event bus.
    """
    def __init__(self):
        self.event_bus: Optional[EventBus] = None

