import tkinter
from tkinter import ttk
from PIL import Image, ImageTk
from structs import Event

class ChipView:
    def __init__(self, parent, event_dispatcher):
        self.frame = ttk.LabelFrame(parent, text="Chip View")
        self.model = event_dispatcher

        self.zoom_level = 1
        self.canvas_width = 400
        self.canvas_height = 400
        self.base_image_size = 100

        # self.path = StringVar()
        self.image_cache = dict()
        
        # Canvas
        self.canvas = tkinter.Canvas(self.frame, width=self.canvas_width, height=self.canvas_height, bg="white", scrollregion=(0, 0, 1000, 1000))
        
        # Scrollbars
        self.hscroll = ttk.Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)
        self.vscroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hscroll.set, yscrollcommand=self.vscroll.set)
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.hscroll.grid(row=1, column=0, sticky="ew")

        self._draw_grid()
        self._draw_image(500, 500)
        self._draw_image(350, 400)
        
        # Function binds
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        self.model.add_event_listener(Event.CHIP_CHANGED, lambda: self._on_chip_changed())

    def _draw_grid(self):
        min_x, max_x, min_y, max_y = 0, 1000, 0, 1000 
        grid_size = 50
        for x in range(int(min_x//grid_size)*grid_size, int(max_x)+grid_size, grid_size):
            self.canvas.create_line(x, min_y, x, max_y, fill="#d0d0d0", tags="grid")
        for y in range(int(min_y//grid_size)*grid_size, int(max_y)+grid_size, grid_size):
            self.canvas.create_line(min_x, y, max_x, y, fill="#d0d0d0", tags="grid")

    def _on_mousewheel(self, event):
        _shift = (event.state & 0x0001) != 0
        _ctrl = (event.state & 0x0004) != 0
        
        direction = 1 if event.delta > 0 else -1

        if _ctrl:
            factor = 1.1 if direction > 0 else 0.9
            self._change_zoom(event, direction)
        elif _shift: self.canvas.xview_scroll(-direction, "units")
        else: self.canvas.yview_scroll(-direction, "units")

    def _change_zoom(self, event, direction):
        factor = 1.1 if direction > 0 else 0.9
        new_zoom = self.zoom_level * factor

        if 0.1 < new_zoom < 10:
            # Get canvas mouse position
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)

            # Apply zoom to all canvas elements
            self.canvas.scale("all", x, y, factor, factor)

            # Update zoom level
            self.zoom_level = new_zoom

            # Rescale images
            self._rescale_all_images()

            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _get_scaled_image(self, path: str):
        try:
            _, thumbnail_img, photo_img, current_scale = self.image_cache[path]
            if abs(current_scale - self.zoom_level) < 0.01:
                return photo_img  # Return cached scaled image
        except KeyError:
            # Load original image only to generate the thumbnail
            original_img = Image.open(path)
            
            # Create a massively downscaled thumbnail (fixed size)
            thumbnail_size = (self.base_image_size, self.base_image_size)
            thumbnail_img = original_img.copy()
            thumbnail_img.thumbnail(thumbnail_size, Image.LANCZOS)
            
            self.image_cache[path] = (original_img, thumbnail_img, None, 1.0)  # Store only thumbnail

        # Calculate new size based on zoom
        original_width, original_height = thumbnail_img.size
        new_size = (
            int(original_width * self.zoom_level),
            int(original_height * self.zoom_level)
        )

        # Create scaled version from the thumbnail
        scaled_img = thumbnail_img.resize(new_size, Image.LANCZOS)
        photo_img = ImageTk.PhotoImage(scaled_img)

        # Update cache
        self.image_cache[path] = (None, thumbnail_img, photo_img, self.zoom_level)  # Store only thumbnail

        return photo_img


    def _rescale_all_images(self):
        for item in self.canvas.find_all():
            if self.canvas.type(item) == "image":
                # Get current image tags (we'll store path in tags)
                tags = self.canvas.gettags(item)
                if tags and len(tags) > 0:
                    path = tags[0]
                    new_photo = self._get_scaled_image(path)
                    self.canvas.itemconfig(item, image=new_photo)

    def _draw_image(self, x=0, y=0, path="stepper_captures/output_2025-03-31_13-57-29.png"):
        """Draw an image that will scale with zoom"""
        photo = self._get_scaled_image(path)
        img_item = self.canvas.create_image(x, y, image=photo, anchor="nw", tags=(path,))
        return img_item
    
    def _on_chip_changed(self):
        exposure = self.model.chip.layers[-1].exposures[-1]
        self._draw_image(*exposure.coords[:2], exposure.path)
