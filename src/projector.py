from typing import Any, Optional
from PIL import Image

from core.engine_module import EngineModule
from core.events import ColorMode, Event, EventBus, ProjectorImageSource


class ProjectorController(EngineModule):
    color_mode: ColorMode = ColorMode.DISABLE
    image_source: ProjectorImageSource = ProjectorImageSource.ACTIVE_LAYER
    custom_image_path: Optional[str] = None
    current_image: Optional[Image.Image] = None
    dlpc: Optional[Any] = None
    project: Optional[Any] = None

    def __init__(self, dlpc: Optional[Any] = None):
        super().__init__()
        self._event_bus: Optional[EventBus] = None
        self.dlpc = dlpc
        self.color_mode = ColorMode.DISABLE
        self.image_source = ProjectorImageSource.ACTIVE_LAYER
        self.custom_image_path = None
        self.current_image = None
        self.project = None

    @property
    def event_bus(self) -> Optional[EventBus]:
        return self._event_bus

    @event_bus.setter
    def event_bus(self, bus: Optional[EventBus]):
        if getattr(self, "_event_bus", None) is not None:
            self._event_bus.remove_listener(Event.PROJECT_CHANGED, self._on_project_changed)
            self._event_bus.remove_listener(Event.ACTIVE_LAYER_CHANGED, self._on_active_layer_changed)
            self._event_bus.remove_listener(Event.ACTIVE_TILE_CHANGED, self._on_active_tile_changed)
            self._event_bus.remove_listener(Event.EXPOSURE_CONFIG_CHANGED, self._on_exposure_config_changed)
        self._event_bus = bus
        if self._event_bus is not None:
            self._event_bus.add_listener(Event.PROJECT_CHANGED, self._on_project_changed)
            self._event_bus.add_listener(Event.ACTIVE_LAYER_CHANGED, self._on_active_layer_changed)
            self._event_bus.add_listener(Event.ACTIVE_TILE_CHANGED, self._on_active_tile_changed)
            self._event_bus.add_listener(Event.EXPOSURE_CONFIG_CHANGED, self._on_exposure_config_changed)

    def _on_project_changed(self, project=None):
        if project is not None:
            self.project = project
        self.update_display()

    def _on_active_layer_changed(self, layer_index=None):
        self.update_display()

    def _on_active_tile_changed(self, tile_index=None):
        self.update_display()

    def _on_exposure_config_changed(self, *args):
        self.update_display()

    def set_project(self, project: Optional[Any]):
        self.project = project
        self.update_display()

    def set_color_mode(self, mode: ColorMode):
        self.color_mode = mode
        if self._event_bus is not None:
            self._event_bus.emit(Event.PROJECTOR_COLOR_MODE_CHANGED, self.color_mode)
        self.update_display()

    def set_image_source(self, source: ProjectorImageSource, custom_path: Optional[str] = None):
        self.image_source = source
        self.custom_image_path = custom_path
        if self._event_bus is not None:
            self._event_bus.emit(Event.PROJECTOR_IMAGE_SOURCE_CHANGED, self.image_source)
        self.update_display()

    def update_display(self):
        """Re-reads the image from the chip project and updates the display.

        ProjectorController is the exclusive emitter of PROJECTOR_IMAGE_CHANGED.
        """
        if self.color_mode == ColorMode.DISABLE:
            self.current_image = None
            self.clear()
            if self.dlpc is not None:
                try:
                    self.dlpc.set_illumination_enable(0)
                except Exception as e:
                    print(f"DLPC LED sync failed: {e}")
        else:
            img = None
            if self.project is not None:
                img = self.project.render_for_projector(
                    color_mode=self.color_mode,
                    image_source=self.image_source,
                    custom_path=self.custom_image_path,
                    projector_size=self.size(),
                )
            self.current_image = img
            if img is not None:
                self.show(img)
            else:
                self.clear()

            if self.dlpc is not None:
                mask = 0b001 if self.color_mode == ColorMode.RED else 0b100
                try:
                    self.dlpc.set_illumination_enable(mask)
                except Exception as e:
                    print(f"DLPC LED sync failed: {e}")

        if self._event_bus is not None:
            self._event_bus.emit(Event.PROJECTOR_IMAGE_CHANGED, self.current_image)

    def show(self, image: Image.Image):
        print("ignoring show image on dummy projector")
        self.clear()

    def size(self) -> tuple[int, int]:
        return (1920, 1080)

    def clear(self):
        print("ignoring clear on dummy projector")
