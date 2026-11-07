import cv2
from abc import ABC, abstractmethod
import threading
import time
from typing import Any, Optional
import numpy as np

from core.engine_module import EngineModule
from core.events import ColorMode, Event, EventBus, ProjectorImageSource
from core.chip_project import ChipProject

class ProjectorController(EngineModule, ABC):
    """Abstract base class defining the projector controller interface.

    Handles image selection, channel filtering, software-generated mask buffers,
    DLPC hardware illumination sync, and display synchronization.
    """



    def __init__(self, dlpc: Optional[Any] = None):
        super().__init__()
        # modes & configs
        self.color_mode: ColorMode = ColorMode.RED
        self.image_source: ProjectorImageSource = ProjectorImageSource.ACTIVE_LAYER
        self.brightness: float = 1.0
        self.is_on: bool = False
        self.custom_image_path: Optional[str] = None
        self.generated_image: Optional[np.ndarray] = None

        self._displayed_image_cache: Optional[np.ndarray] = None
        self.dlpc = dlpc

        self.project: Optional[ChipProject] = None
        self.display_ready_event = threading.Event()
        self.display_ready_event.set()

    def _on_detach_event_bus(self, bus: EventBus):
        bus.remove_listener(Event.PROJECT_CHANGED, self._on_project_changed)
        bus.remove_listener(Event.ACTIVE_LAYER_CHANGED, self._on_active_layer_changed)
        bus.remove_listener(Event.ACTIVE_TILE_CHANGED, self._on_active_tile_changed)
        bus.remove_listener(Event.LAYER_CACHE_RECOMPUTED, self._on_layer_cache_recomputed)

    def _on_attach_event_bus(self, bus: EventBus):
        bus.add_listener(Event.PROJECT_CHANGED, self._on_project_changed)
        bus.add_listener(Event.ACTIVE_LAYER_CHANGED, self._on_active_layer_changed)
        bus.add_listener(Event.ACTIVE_TILE_CHANGED, self._on_active_tile_changed)
        bus.add_listener(Event.LAYER_CACHE_RECOMPUTED, self._on_layer_cache_recomputed)



    def _on_project_changed(self, project=None):
        if project is not None:
            self.project = project
        self._recompute_image()
        self.update_display()

    def _on_active_layer_changed(self, layer_index=None):
        self._recompute_image()
        self.update_display()

    def _on_active_tile_changed(self, tile_index=None):
        self._recompute_image()
        self.update_display()

    def _on_layer_cache_recomputed(self, layer=None, *args):
        self._recompute_image()
        self.update_display()


    def set_project(self, project: Optional[Any]):
        self.project = project
        self._recompute_image()
        self.update_display()

    def set_color_mode(self, mode: ColorMode):
        self.color_mode = mode
        if self.event_bus is not None:
            self.event_bus.emit(Event.PROJECTOR_COLOR_MODE_CHANGED, self.color_mode)
        self._recompute_image()
        self.update_display()

    def set_image_source(self, source: ProjectorImageSource, custom_path: Optional[str] = None):
        self.image_source = source
        self.custom_image_path = custom_path
        if self.event_bus is not None:
            self.event_bus.emit(Event.PROJECTOR_IMAGE_SOURCE_CHANGED, self.image_source)
        self._recompute_image()
        self.update_display()

    def set_generated_image(self, image: Optional[np.ndarray]):
        """Sets the software-generated image buffer."""
        self.generated_image = image
        if self.image_source == ProjectorImageSource.GENERATED:
            self._recompute_image()
            self.update_display()

    def set_brightness(self, brightness: float):
        """Sets the projector image brightness multiplier (clamped to [0.0, 1.0])."""
        self.brightness = max(0.0, min(1.0, float(brightness)))
        if self.event_bus is not None:
            self.event_bus.emit(Event.PROJECTOR_BRIGHTNESS_CHANGED, self.brightness)
        self._recompute_image()
        self.update_display()




    def _recompute_image(self):
        """Precomputes and caches the rendered pattern based on image source and color mode.

        All image calculations are performed here so that toggling the projector on/off
        is instantaneous with zero processing overhead.
        """
        # get the image
        img: Optional[np.ndarray] = None
        if self.image_source == ProjectorImageSource.SOLID:
            proj_size = self.projector_size()
            img = np.full((proj_size[1], proj_size[0], 3), 255, dtype=np.uint8)

        elif self.image_source == ProjectorImageSource.GENERATED:
            img = self.generated_image

        elif self.image_source == ProjectorImageSource.CUSTOM_FILE:
            if self.custom_image_path is not None:
                loaded_img = cv2.imread(self.custom_image_path)
                if loaded_img is None:
                    print("Failed to load custom image")
                else:
                    proj_w, proj_h = self.projector_size()
                    img_h, img_w = loaded_img.shape[:2]
                    if proj_w > 0 and proj_h > 0 and img_w > 0 and img_h > 0:
                        scale = min(proj_w / img_w, proj_h / img_h)
                        new_w = max(1, int(round(img_w * scale)))
                        new_h = max(1, int(round(img_h * scale)))
                        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                        resized = cv2.resize(loaded_img, (new_w, new_h), interpolation=interp)

                        if resized.ndim == 2:
                            canvas = np.zeros((proj_h, proj_w), dtype=resized.dtype)
                        else:
                            canvas = np.zeros((proj_h, proj_w, resized.shape[2]), dtype=resized.dtype)

                        x_offset = (proj_w - new_w) // 2
                        y_offset = (proj_h - new_h) // 2
                        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
                        img = canvas
                    else:
                        img = loaded_img

        elif self.image_source == ProjectorImageSource.ACTIVE_LAYER and self.project is not None:
            tile = self.project.active_tile
            if tile is not None:
                img = tile[0]

        # apply color mode
        if img is not None:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = img.copy()
            if self.color_mode == ColorMode.RED:
                img[:, :, 0:2] = 0
            elif self.color_mode == ColorMode.UV:
                img[:, :, 1:3] = 0

            # Apply brightness adjustment for red only
            if self.color_mode == ColorMode.RED and self.brightness < 1.0:
                img = np.clip(np.round(img.astype(np.float32) * self.brightness), 0, 255).astype(img.dtype)

        self._displayed_image_cache = img
        self._on_display_image_cache_changed()


    def set_on(self, on: bool):
        """Switches the projector illumination output on or off.

        Does not perform any image calculations; switches immediately between
        blank (black) and the precomputed prepared_image.
        """
        self.is_on = on

        if self.dlpc is not None:
            mask = (0b001 if self.color_mode == ColorMode.RED else 0b100) if self.is_on else 0
            try:
                self.dlpc.set_illumination_enable(mask)
            except Exception as e:
                print(f"DLPC LED sync failed: {e}")

        if self.event_bus is not None:
            self.event_bus.emit(Event.PROJECTOR_ON_OFF_CHANGED, self.is_on)

        self.update_display()

    def _on_display_image_cache_changed(self):
        """Override this method to run custom logic when display image changed."""
        if self.event_bus is not None:
            self.event_bus.emit(Event.PROJECTOR_IMAGE_CHANGED, self._displayed_image_cache)

    @abstractmethod
    def update_display(self):
        """Updates display state and emits PROJECTOR_IMAGE_CHANGED.

        ProjectorController is the exclusive emitter of PROJECTOR_IMAGE_CHANGED.
        """

    def wait_for_display(self, timeout: float = 2.0) -> bool:
        """Blocks until the display window has completed rendering the current frame."""
        if threading.current_thread() is threading.main_thread():
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    app.processEvents()
            except ImportError:
                pass
            return True
        return self.display_ready_event.wait(timeout=timeout)

    @abstractmethod
    def projector_size(self) -> tuple[int, int]:
        """Projector display resolution (width, height) in pixels."""
        pass


class DummyProjector(ProjectorController):
    """Simulated projector controller for headless environments, testing, and development."""

    def __init__(self, size: tuple[int, int] = (1920, 1080), dlpc: Optional[Any] = None):
        self._size = size
        super().__init__(dlpc=dlpc)

    def projector_size(self) -> tuple[int, int]:
        return self._size

    def update_display(self):
        pass
