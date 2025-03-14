from tkinter import Label, Tk, Toplevel

from PIL import Image, ImageTk

from lib.img import image_to_tk_image


class ProjectorController:
    def show(self, image: Image.Image):
        print("ignoring show image on dummy projector")
        self.clear()

    def size(self) -> tuple[int, int]:
        return (1920, 1080)

    def clear(self):
        print("ignoring clear on dummy projector")


# creates a fullscreen window and displays specified images to it
class TkProjector(ProjectorController):
    ### Internal Fields ###
    window: Toplevel
    label: Label

    # just a black image to clear with
    clear_image: ImageTk.PhotoImage

    def __init__(self, root: Tk, title: str = "Projector", background: str = "#000000"):
        self.window = Toplevel(root)
        self.window.title(title)
        self.window.attributes("-fullscreen", True)
        self.window["background"] = background
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        # create projection Label
        self.label = Label(self.window, bg="black")
        self.label.grid(row=0, column=0, sticky="nesw")

        # generate dummy black image
        self.clear_image = image_to_tk_image(Image.new("L", self.size()))

    def size(self) -> tuple[int, int]:
        return (self.window.winfo_width(), self.window.winfo_height())

    # show an image
    # if a duration is specified, show the image for that many milliseconds
    # Calls update_func during patterning with a single argument from 0.0-1.0 indicating progress
    def show(self, image: Image.Image):
        # if(self.__is_patterning__):
        #  if(self.debug != None):
        #    self.debug.warn("Tried to show image while another is still showing")
        # return False
        # warn if image isn't correct size
        # if(image.size != fit_image(image, self.size())):
        #  if(self.debug != None):
        #    self.debug.warn("projecting image with incorrect size:\n  "+str(image.size)+" instead of "+str(self.size()))
        photo = image_to_tk_image(image)
        self.label.configure(image=photo)  # type:ignore
        self.photo = photo

    def clear(self):
        self.label.configure(image=self.clear_image)  # type:ignore
