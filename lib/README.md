# Main library for hackerfab GUI and related code

## Backend_lib
primarily used to support the `img_lib`, it contains a few highly useful classes.

### Smart_Image
A widget type that stores an image with a few helper functions to assist in image modification and resetting. 

Primarily, allows adding attributes / flags to an image and modifying said image while providing a way to reset to previous versions of that image. 

There is Intentionally no way to change the original image. Each instance of this class should be treated as an image, and created / deleted accordingly.

This is especially useful when applying filters, like toggling color channels, on images for patterning.

### Stage_Controller

A highly generic module for managing the location, movement, and hardware updating of a stage. 

Operates by a user adding callbacks to a stage which are called when th stage is moved.

It also allows user calibration of a stage so click-to-move functionality can be easily implemented. No CV necessary.

See `Lithographer.py` for a hardware linked example, and a image adjustment software-only example. 

### Multi_Stage

A convenient way to manage multiple stage classes. 

For the probe station, each probe is its own 'stage' and this controls the lot of them.

### Func_Manager

A convenient way to tie many calls to a single function and allow easy enabling and disabling.

## gui_lib

User facing widgets and windows to allow easier and more advanced use of TKinter. 

### Debug

Links to `GUI_Controller` and allows a convenient way to display info, warnings, and errors to the UI. 

### Cycle

Similar to a toggle button, but allows an endless number of states. 

Has a handful of convenient methods like changing button colors, text, etc for each state. 

Used to cycle between `Smart_Area`s, cycle between color channels, toggle a simple action on or off. 

### Thumbnail

An easy way to prompt the user for images, store, and display them in a UI. 

Used to import images to the lLithographer UI.

### Intput

Similar to a numeric or text input, but specifically for integers. 

Unlike the default TKinter variable fields, has several convenience functions like limits, steps, and checking for changes since last query. 

### Floatput

Same as an Intput, but for floats. 

### Porjector_Controller

Linked to a `GUI_Controller`; handles generating and updating a fullscreen window to use on the lithographer projector

### TextPopup

Creates a new popup window with some text. Useful for help menus and tutorials.

TODO: overhaul to allow for a more interactive window: smart areas, images, etc

### GUI_Controller

Manages the debug and projector widget, as well as allows storing of all widgets in a UI for easy access in the code. 

Main benefit of storing all widgets within, is they can be dynamically accessed within code by addressing widgets with strings. 

### Smart_Area

Handles swapping between sets of widgets in an area in a UI. 

Within `Lithographer.py`, is used for showing and hiding the settings pane, as well as toggling between stage and fine adjustment controls. 

## img_lib

A myriad of miscellaneous image and image related functions.

Because there are so many, only the most important are included here.

### toggle_channels

Return a new image with specified color channels enabled/disables

### fit, fill, center crop

Return new dimensions for an image to fit / fill a window or area or crop into an image's center.

### convert_to_alpha_mask

Converts an image to an alpha mask where opacity is scaled to the image's color scale. 

Used in `Lithographer.py` to generate flatfield correction masks from images of a blank pattern.

### posterize

flattens an image to only pure white or pure black pixels. Useful for cleaning up edges on patterns.

Includes a threashold for what should be considered black or white.

### Better Transform

A much, much simpler way to generate and apply affine transformations to an image.

### Tuple Operations

Quick functions to do arithmetic on tuples.

Most support int expansion: add((1,2,3),1) -> (2,3,4)

