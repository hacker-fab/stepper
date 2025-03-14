# Hacker Fab
# Luca Garlati, 2024
# GUI library for hackerfab UI scripts

# region: imports
from __future__ import annotations

from tkinter import (
    BooleanVar,
    Button,
    DoubleVar,
    Entry,
    IntVar,
    Label,
    Tk,
    Toplevel,
    Variable,
    Widget,
    filedialog,
)
from typing import Any, Callable, Literal, Optional, Union

from PIL import Image, ImageTk

# import sys and use path insert to add lib files
from .img import LA_to_L, RGBA_to_RGB, fit_image, image_to_tk_image
from .tuple import *

# endregion


# widget to display info, errors, warning, and text
class Debug:
    __widget__: Label
    __enabled__: bool
    text_color: tuple[str, str]
    warn_color: tuple[str, str]
    err_color: tuple[str, str]

    # create new widget
    def __init__(
        self,
        root: Tk,
        text_color: tuple[str, str] = ("black", "white"),
        warn_color: tuple[str, str] = ("black", "orange"),
        err_color: tuple[str, str] = ("white", "red"),
    ):
        self.__enabled__ = True
        self.text_color = text_color
        self.warn_color = warn_color
        self.err_color = err_color
        self.__widget__ = Label(root, justify="left", anchor="w")
        self.__set_color__(text_color)

    # show text in the debug widget
    def info(self, text: str):
        if not self.__enabled__:
            return
        self.__widget__.config(text=text)
        self.__set_color__(self.text_color)
        print("i " + text)

    # show warning in the debug widget
    def warn(self, text: str):
        if not self.__enabled__:
            return
        self.__widget__.config(text=text)
        self.__set_color__(self.warn_color)
        print("w " + text)

    # show error in the debug widget
    def error(self, text: str):
        if not self.__enabled__:
            return
        self.__widget__.config(text=text)
        self.__set_color__(self.err_color)
        print("e " + text)

    # enable prints
    def enable(self):
        self.__enabled__ = True

    # disable prints
    def disable(self):
        self.__enabled__ = False

    # place widget on the grid
    def grid(
        self,
        row: int | None = None,
        col: int | None = None,
        colspan: int = 1,
        rowspan: int = 1,
    ):
        if row == None or col == None:
            self.__widget__.grid()
        else:
            self.__widget__.grid(
                row=row, column=col, rowspan=rowspan, columnspan=colspan, sticky="nesw"
            )

    # remove widget from the grid
    def grid_remove(self):
        self.__widget__.grid_remove()

    # set the text and background color
    def __set_color__(self, colors: tuple[str, str]):
        self.__widget__.config(fg=colors[0], bg=colors[1])


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
                    img = RGBA_to_RGB(img)
                    # if(self.__gui__.debug != None):
                    #  self.__gui__.debug.warn("RGBA images are not permitted, auto converted to RGB")
            case "LA":
                if not self.accept_alpha:
                    img = LA_to_L(img)
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


# creates a better int input field
class Intput:
    __widget__: Entry
    __gui__: GUI_Controller
    __total_intputs__: int = 0
    var: Variable
    # user fields
    min: int | None
    max: int | None
    name: str
    invalid_color: str
    # revert displayed value to last valid value if invalid?
    auto_fix: bool
    # optional validation function, true if input is valid
    extra_validation: Callable[[int], bool] | None
    # the value that will be returned: always valid
    __value__: int
    # value checked by changed()
    last_diff: int

    def __init__(
        self,
        gui: GUI_Controller,
        parent: Any | None = None,
        name: str | None = None,
        default: int = 0,
        min: int | None = None,
        max: int | None = None,
        justify: Literal["left", "center", "right"] = "center",
        extra_validation: Callable[[int], bool] | None = None,
        auto_fix: bool = True,
        invalid_color: str = "red",
    ):
        # store user inputs
        self.__gui__ = gui
        self.min = min
        self.max = max
        self.extra_validation = extra_validation
        self.auto_fix = auto_fix
        self.invalid_color = invalid_color
        # setup var
        self.var = IntVar()
        self.var.set(default)
        self.value = self.min
        self.last_diff = default
        # setup widget
        self.__widget__ = Entry(
            parent if parent is not None else gui.root,
            textvariable=self.var,
            justify=justify,
        )
        # set name
        if name == None:
            self.name = "unnamed intput widget " + str(Intput.__total_intputs__)
            Intput.__total_intputs__ += 1
            gui.add_widget(self.name, self)
        else:
            gui.add_widget(name, self)
        # update
        self.__update__()

    # place widget on the grid
    def grid(
        self,
        row: int | None = None,
        col: int | None = None,
        colspan: int = 1,
        rowspan: int = 1,
    ):
        if row == None or col == None:
            self.__widget__.grid()
        else:
            self.__widget__.grid(
                row=row, column=col, rowspan=rowspan, columnspan=colspan, sticky="nesw"
            )

    # remove widget from the grid
    def grid_remove(self):
        self.__widget__.grid_remove()

    # get the more recent vaid value
    def get(self, update: bool = True) -> int:
        if update:
            self.__update__()
        return self.__value__

    # try and set a new value
    def set(self, user_value: int):
        self.__update__(user_value)

    # has the value changed since the last time this method was called
    def changed(self) -> bool:
        if self.get() != self.last_diff:
            self.last_diff = self.get()
            return True
        return False

    # updates widget and value
    def __update__(self, new_value: int | None = None):
        # get new potential value
        new_val: int
        if new_value == None:
            new_val = self.var.get()
        else:
            new_val = new_value
        # validate and update accordingly
        if self.__validate__(new_val):
            self.__value__ = new_val
            self.var.set(new_val)
            self.__widget__.config(bg="white")
        else:
            if self.auto_fix:
                self.var.set(self.__value__)
            else:
                self.__widget__.config(bg=self.invalid_color)
            if self.__gui__.debug != None:
                self.__gui__.debug.error(
                    "Invalid value for " + self.name + ": " + str(new_val)
                )
        self.__widget__.update()

    # check if the current value is valid
    def __validate__(self, new_val: int) -> bool:
        # check min / max
        if self.min != None and new_val < self.min:
            return False
        if self.max != None and new_val > self.max:
            return False
        # check extra validation
        if self.extra_validation != None and not self.extra_validation(new_val):
            return False
        # passed all checks
        return True


# creates a better float input field
class Floatput:
    # private fields
    __widget__: Entry
    __gui__: GUI_Controller
    __total_floatputs__: int = 0
    accuracy: int
    var: Variable
    # user fields
    min: float | None
    max: float | None
    name: str
    invalid_color: str
    # revert displayed value to last valid value if invalid
    auto_fix: bool
    # optional validation function, true if input is valid
    extra_validation: Callable[[float], bool] | None
    # the value that will be returned: always valid
    __value__: float
    # value checked by changed()
    last_diff: float

    def __init__(
        self,
        gui: GUI_Controller,
        name: str | None = None,
        default: float = 0,
        display_accuracy: int = 2,
        min: float | None = None,
        max: float | None = None,
        justify: Literal["left", "center", "right"] = "center",
        extra_validation: Callable[[float], bool] | None = None,
        auto_fix: bool = True,
        invalid_color: str = "red",
    ):
        # store user inputs
        self.__gui__ = gui
        self.accuracy = display_accuracy
        self.min = min
        self.max = max
        self.extra_validation = extra_validation
        self.auto_fix = auto_fix
        self.invalid_color = invalid_color
        # setup var
        self.var = DoubleVar()
        self.var.set(default)
        self.value = self.min
        self.last_diff = default
        # setup widget
        self.__widget__ = Entry(gui.root, textvariable=self.var, justify=justify)
        # set name
        if name == None:
            self.name = "unnamed floatput widget " + str(floatput.__total_floatputs__)
            floatput.__total_floatputs__ += 1
            gui.add_widget(self.name, self)
        else:
            gui.add_widget(name, self)
        # update
        self.__update__()

    # place widget on the grid
    def grid(
        self,
        row: int | None = None,
        col: int | None = None,
        colspan: int = 1,
        rowspan: int = 1,
    ):
        if row == None or col == None:
            self.__widget__.grid()
        else:
            self.__widget__.grid(
                row=row, column=col, rowspan=rowspan, columnspan=colspan, sticky="nesw"
            )

    # remove widget from the grid
    def grid_remove(self):
        self.__widget__.grid_remove()

    # get the more recent vaid value
    def get(self, update: bool = True) -> float:
        if update:
            self.__update__()
        return self.__value__

    # try and set a new value
    def set(self, user_value: float):
        self.__update__(user_value)

    # has the value changed since the last time this method was called
    def changed(self) -> bool:
        if self.get() != self.last_diff:
            self.last_diff = self.get()
            return True
        return False

    # updates widget and value
    def __update__(self, new_value: float | None = None):
        # get new potential value
        new_val: float
        if new_value == None:
            new_val = self.var.get()
        else:
            new_val = new_value
        # validate and update accordingly
        if self.__validate__(new_val):
            self.__value__ = new_val
            self.var.set(round(new_val, self.accuracy))
            self.__widget__.config(bg="white")
        else:
            if self.auto_fix:
                self.var.set(self.__value__)
            else:
                self.__widget__.config(bg=self.invalid_color)
            if self.__gui__.debug != None:
                self.__gui__.debug.error(
                    "Invalid value for " + self.name + ": " + str(new_val)
                )
        self.__widget__.update()

    # check if the current value is valid
    def __validate__(self, new_val: float) -> bool:
        # check min / max
        if self.min != None and new_val < self.min:
            return False
        if self.max != None and new_val > self.max:
            return False
        # check extra validation
        if self.extra_validation != None and not self.extra_validation(new_val):
            return False
        # passed all checks
        return True


# creates a new window with specified text. Useful for a help menu
class TextPopup:
    ### Internal Fields ###
    __TL__: Toplevel
    __label__: Label
    __root__: Tk
    __widget__: Button
    ### User Fields ###
    button_text: str
    popup_text: str
    debug: Debug | None = None

    def __init__(
        self,
        root: Tk,
        button_text: str = "",
        popup_text: str = "",
        title: str = "Popup",
        debug: Debug | None = None,
    ):
        # assign vars
        self.__root__ = root
        self.button_text = button_text
        self.popup_text = popup_text
        self.title = title
        self.debug = debug
        # build button widget
        button: Button = Button(root, command=self.show)
        if button_text != "":
            button.config(text=button_text, compound="top")
        self.__widget__ = button

    # show the text popup
    def show(self):
        self.__TL__ = Toplevel(self.__root__)
        self.__TL__.title(self.title)
        self.__TL__.grid_columnconfigure(0, weight=1)
        self.__TL__.grid_rowconfigure(0, weight=1)
        self.__label__ = Label(self.__TL__, text=self.popup_text, justify="left")
        self.__label__.grid(row=0, column=0, sticky="nesw")
        if self.debug != None:
            self.debug.info("Showing " + self.button_text + " popup")
        self.update()

    # place widget on the grid
    def grid(
        self,
        row: int | None = None,
        col: int | None = None,
        colspan: int = 1,
        rowspan: int = 1,
    ):
        if row == None or col == None:
            self.__widget__.grid()
        else:
            self.__widget__.grid(
                row=row, column=col, rowspan=rowspan, columnspan=colspan, sticky="nesw"
            )

    # remove widget from the grid
    def grid_remove(self):
        self.__widget__.grid_remove()

    def update(self, new_text: str = ""):
        if new_text != "":
            self.text = new_text
            self.__label__.config(text=new_text)
        self.__root__.update()
        self.__TL__.update()


# TODO auto adjust rows and cols when adding children
# TODO auto debug widget creation and application to children
# GUI controller and widget manager
class GUI_Controller:
    # list of accepted widget types
    gui_widgets = Union[Widget, Thumbnail, Debug, Intput, Floatput, TextPopup]
    # region: fields
    ### Internal Fields ###
    root: Tk
    __widgets__: dict[str, gui_widgets]
    ### User Fields ###
    # Mandatory
    grid_size: tuple[int, int]
    # Optional
    title: str
    window_size: tuple[int, int]
    resizeable: bool
    debug: Debug | None = None
    colors: dict[str, tuple[str, str]]
    # endregion

    def __init__(
        self,
        grid_size: tuple[int, int],
        set_window_size: tuple[int, int] = (0, 0),
        add_window_size: tuple[int, int] = (0, 0),
        title: str = "GUI Controller",
        resizeable: bool = True,
    ):
        # store user input variables
        self.grid_size = grid_size
        self.title = title
        self.resizeable = resizeable
        # setup root / gui window
        self.root = Tk()
        self.window_size = set_window_size
        if set_window_size == (0, 0):
            self.window_size = (
                self.root.winfo_screenwidth() // 2,
                self.root.winfo_screenheight() // 2,
            )
        if add_window_size != (0, 0):
            self.window_size = (
                self.window_size[0] + add_window_size[0],
                self.window_size[1] + add_window_size[1],
            )
        self.root.title(self.title)
        self.root.geometry(str(self.window_size[0]) + "x" + str(self.window_size[1]))
        self.root.resizable(width=self.resizeable, height=self.resizeable)
        for row in range(self.grid_size[0]):
            self.root.grid_rowconfigure(row, weight=1)
        for col in range(self.grid_size[1]):
            self.root.grid_columnconfigure(col, weight=1)
        # create dictionary of widgets
        self.__widgets__ = {}

    def add_widget(self, name: str, widget: gui_widgets):
        # if a debug widget is added, save it as the debug field
        if type(widget) == Debug:
            self.debug = widget
        else:
            self.__widgets__[name] = widget
        self.update()

    # return widget by name, or None if not found
    def get_widget(self, name: str) -> gui_widgets | None:
        return self.__widgets__.get(name, None)

    # remove widget by name
    def del_widget(self, name: str):
        # remove widget from dictionary
        widget = self.__widgets__.pop(name, None)
        # check if widget was found
        if widget == None:
            if self.debug != None:
                self.debug.warn(
                    "Tried to remove widget " + name + " but it was not found"
                )
            return
        # report success
        self.debug.info("Removed widget " + name)
        self.update()

    # update the GUI window
    def update(self):
        self.root.update()

    # start the main loop
    def mainloop(self):
        self.root.mainloop()

    # get location within a widget as percentage from top left
    # if no widget specified, or widget not found, return location within the root window
    # if img_size is specified, return location within image
    # if in_pixels, return pixel location instead of percentage
    def get_coords(
        self,
        widget: str = "",
        img_size: tuple[int, int] = (0, 0),
        in_pixels: bool = False,
    ) -> tuple[float, float] | tuple[int, int]:
        # get widget
        this_widget = self.get_widget(widget)
        if this_widget == None:
            widget = "root window"
            if self.debug != None:
                self.debug.warn("Using root window for get_coords()")
            this_widget = self.root
        widget_size = (this_widget.winfo_width(), this_widget.winfo_height())
        # get location within *WINDOW*
        button_pressed = BooleanVar()
        coords: tuple[int, int] = (0, 0)

        def __get_coords__(event) -> tuple:
            nonlocal coords
            coords = (event.x, event.y)
            button_pressed.set(True)
            this_widget.unbind("<Button 1>")

        this_widget.bind("<Button 1>", __get_coords__)
        this_widget.wait_variable(button_pressed)
        # offset by widget location
        result = coords
        # result = sub(coords, (this_widget.winfo_x(), this_widget.winfo_y()))
        if (result[0] < 0 or result[1] < 0) and self.debug != None:
            self.debug.warn("Clicked outside of " + widget)
        # if image size is specified, return location within image
        if img_size != (0, 0):
            result = sub(result, div(sub(widget_size, img_size), 2))
            if in_pixels:
                return result
            else:
                return div(result, img_size)
        else:
            if in_pixels:
                return result
            else:
                return div(result, widget_size)
