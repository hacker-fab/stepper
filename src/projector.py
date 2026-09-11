from typing import Any, Optional
from PIL import Image

from core.engine_module import EngineModule
from core.events import Event, ShownImage


class ProjectorController(EngineModule):
    mode: ShownImage = ShownImage.CLEAR
    current_image: Optional[Image.Image] = None
    dlpc: Optional[Any] = None

    def __init__(self, dlpc: Optional[Any] = None):
        super().__init__()
        self.dlpc = dlpc
        self.mode = ShownImage.CLEAR
        self.current_image = None

    def show(self, image: Image.Image):
        print("ignoring show image on dummy projector")
        self.clear()

    def size(self) -> tuple[int, int]:
        return (1920, 1080)

    def clear(self):
        print("ignoring clear on dummy projector")

    def set_mode(self, mode: ShownImage, image: Optional[Image.Image] = None):
        self.mode = mode
        if mode == ShownImage.CLEAR:
            self.current_image = None
            self.clear()
            if self.dlpc is not None:
                try:
                    self.dlpc.set_illumination_enable(0)
                except Exception as e:
                    print(f"DLPC LED sync failed: {e}")
        else:
            self.current_image = image
            if image is not None:
                self.show(image)
            else:
                self.clear()
            if self.dlpc is not None:
                if mode == ShownImage.RED_FOCUS:
                    mask = 0b001
                elif mode in (ShownImage.UV_FOCUS, ShownImage.PATTERN):
                    mask = 0b100
                else:
                    mask = 0b111
                try:
                    self.dlpc.set_illumination_enable(mask)
                except Exception as e:
                    print(f"DLPC LED sync failed: {e}")

        if self.event_bus is not None:
            self.event_bus.emit(Event.PROJECTOR_IMAGE_CHANGED, self.mode)
