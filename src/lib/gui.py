# Hacker Fab
# Luca Garlati, 2024
# GUI library for hackerfab UI scripts

# region: imports
from __future__ import annotations

from tkinter import (
    Button,
    DoubleVar,
    Entry,
    IntVar,
    Variable,
    filedialog,
)
from typing import Callable, Literal, Optional

from PIL import Image, ImageTk

from .img import fit_image, image_to_tk_image

# endregion


# creates thumbnail / image import widget
class Thumbnail:
    widget: Button
    # image stuff
    image: Image.Image
    path: str
    thumb_image: ImageTk.PhotoImage
    thumb_size: tuple[int, int]
    # optional fields
    text: str
    accept_alpha: bool
    func_on_success: Callable | None

    def __init__(
        self,
        parent,
        thumb_size: tuple[int, int],
        text: str = "",
        accept_alpha: bool = False,
        on_import: Callable | None = None,
    ):
        # assign vars
        self.thumb_size = thumb_size
        self.text = text
        self.accept_alpha = accept_alpha
        self.on_import = on_import
        # build widget
        self.widget = Button(
            parent, text=self.text, command=self._import_image, compound="top"
        )

        self.thumb_size = thumb_size
        # create placeholder images
        self.image = Image.new("RGB", self.thumb_size)
        self.path = ""
        self._refresh()

    # prompt user for a new image
    def _import_image(self):
        def is_valid_ext(path: str) -> bool:
            path = path.lower()
            return (
                path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png")
            )

        # get image
        path: str = filedialog.askopenfilename(title="Open")

        # TODO: Debug
        """
    if self._gui.debug is not None:
      if(path == ''):
        self.__gui__.debug.warn(self.text+(" " if self.text!="" else "")+"import cancelled")
        return
      if(not is_valid_ext(path)):
        self.__gui__.debug.error(self.text+(" " if self.text!="" else "")+"invalid file type: "+path[-3:])
        return
      else:
        self.__gui__.debug.info(self.text+(" " if self.text!="" else "")+"set to "+basename(path))
    """

        img = Image.open(path).copy()
        # check type
        # ensure image is RGB or L
        match img.mode:
            case "RGB":
                pass
            case "L":
                pass
            case "RGBA":
                if not self.accept_alpha:
                    img = img.convert("RGB")
                    # if(self.__gui__.debug != None):
                    #  self.__gui__.debug.warn("RGBA images are not permitted, auto converted to RGB")
            case "LA":
                if not self.accept_alpha:
                    img = img.convert("L")
                    # if(self.__gui__.debug != None):
                    #  self.__gui__.debug.warn("LA images are not permitted, auto converted to L")
            case _:
                # if(self.__gui__.debug != None):
                #  self.__gui__.debug.error("Invalid image mode: "+img.mode)
                return

        # update
        self.image = img
        self.path = path
        self._refresh()

        # call optional func if specified
        if self.on_import is not None:
            (self.on_import)(self)

    def _refresh(self):
        new_size = fit_image(self.image.size, self.thumb_size)
        self.thumb_image = image_to_tk_image(self.image.resize(new_size))
        self.widget.configure(image=self.thumb_image)  # type:ignore


class FloatEntry:
    widget: Entry
    _var: Variable

    def __init__(
        self,
        parent,
        default: int = 0,
        var: Optional[Variable] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        justify: Literal["left", "center", "right"] = "center",
    ):
        self._var = var if var is not None else DoubleVar()
        self._var.set(default)

        self.default = default

        self.min_value = min_value
        self.max_value = max_value

        self.widget = Entry(parent, textvariable=self._var, justify=justify)

        self.widget.bind("<FocusOut>", self._round_display)
        self.widget.bind("<Return>", self._round_display)

    def _round_display(self, event=None):
        try:
            value = float(self.widget.get())
            value = round(value, 1)
            self._var.set(f"{value:.1f}")
        except ValueError:
            self._var.set(f"{self.default:.1f}")

    def get(self) -> float:
        if self.widget.get() == "":
            self.default = round(self.default, 1)
            return self.default
        else:
            return round(self._var.get(), 1)

    def set(self, value: float):
        value = round(value, 1)
        self._var.set(value)


# TODO:
class IntEntry:
    widget: Entry
    _var: Variable

    def __init__(
        self,
        parent,
        default: int = 0,
        var: Optional[Variable] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        justify: Literal["left", "center", "right"] = "center",
    ):
        self._var = var if var is not None else IntVar()
        self._var.set(default)

        self.default = default

        self.min_value = min_value
        self.max_value = max_value

        self.widget = Entry(parent, textvariable=self._var, justify=justify)

    def get(self) -> int:
        if self.widget.get() == "":
            return self.default
        else:
            return self._var.get()

    def set(self, value: int):
        self._var.set(value)
