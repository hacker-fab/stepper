from tkinter import *
from tkinter import ttk
import tkinter
from dataclasses import dataclass

@dataclass
class ButtonStyle:
    text: str
    fg: str
    #bg does not change on MacOS
    #bg: str

class ToggleButton:
    is_enabled: bool

    enabled_style: ButtonStyle
    disabled_style: ButtonStyle

    def __init__(self, parent, enabled_style: ButtonStyle, disabled_style: ButtonStyle, is_enabled=False):
        self.widget = tkinter.Button(parent, command=self._on_click)
        self.is_enabled = is_enabled
        self.enabled_style = enabled_style
        self.disabled_style = disabled_style
        self._update_style()

    def _on_click(self):
        self.is_enabled = not self.is_enabled
        self._update_style()
    
    def _update_style(self):
        style = self.enabled_style if self.is_enabled else self.disabled_style
        self.widget.config(text=style.text, fg=style.fg)

class StagePositionFrame:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)

        COORDS = ('X', 'Y', 'Z')
        for i in range(3):
            coord_entry = ttk.Entry(self.frame)
            coord_inc_button = ttk.Button(self.frame, text=f'+{COORDS[i]}')
            coord_dec_button = ttk.Button(self.frame, text=f'-{COORDS[i]}')

            coord_entry.grid(row=0, column=i)
            coord_inc_button.grid(row=1, column=i)
            coord_dec_button.grid(row=2, column=i)
        
        set_position_button = ttk.Button(self.frame, text='Set Stage Position')
        set_position_button.grid(row=3, column=0, columnspan=3, sticky='ew')

        ttk.Label(self.frame, text='Stage Step Size (microns)', anchor='center').grid(row=4, column=0, columnspan=3, sticky='ew')
        for i in range(3):
            coord_entry = ttk.Entry(self.frame)
            coord_entry.grid(row=5, column=i, sticky='ew')

class FineAdjustFrame:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)

        COORDS = ('x', 'y', 'theta')
        for i in range(3):
            coord_entry = ttk.Entry(self.frame)
            coord_inc_button = ttk.Button(self.frame, text=f'+{COORDS[i]}')
            coord_dec_button = ttk.Button(self.frame, text=f'-{COORDS[i]}')

            coord_entry.grid(row=0, column=i)
            coord_inc_button.grid(row=1, column=i)
            coord_dec_button.grid(row=2, column=i)

        set_position_button = ttk.Button(self.frame, text='Set Fine Adjust')
        set_position_button.grid(row=3, column=0, columnspan=3, sticky='ew')

        ttk.Label(self.frame, text='Fine Adjustment Step Size', anchor='center').grid(row=4, column=0, columnspan=3, sticky='ew')
        for i in range(3):
            coord_entry = ttk.Entry(self.frame)
            coord_entry.grid(row=5, column=i, sticky='ew')




class OptionsFrame:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)

        ttk.Label(self.frame, text='Exposure Time (ms)').grid(row=0, column=0)
        self.exposure_time_entry = ttk.Entry(self.frame)
        self.exposure_time_entry.grid(row=0, column=1, columnspan=3, sticky='ew')

        ttk.Label(self.frame, text='Tiles (horiz, vert)').grid(row=1, column=0)
        self.tiles_horiz_entry = ttk.Entry(self.frame)
        self.tiles_horiz_entry.grid(row=1, column=1)
        self.tiles_vert_entry = ttk.Entry(self.frame)
        self.tiles_vert_entry.grid(row=1, column=2)
        self.tiles_snake_button = ttk.Button(self.frame, text='Snake')
        self.tiles_snake_button.grid(row=1, column=3, sticky='ew')

        ttk.Label(self.frame, text='Flatfield Strength (%)').grid(row=2, column=0)
        self.flatfield_strength_entry = ttk.Entry(self.frame)
        self.flatfield_strength_entry.grid(row=2, column=2)
        self.flatfield_button = ttk.Button(self.frame, text='NOT Using Flatfield')
        self.flatfield_button.grid(row=2, column=3, sticky='ew')

        ttk.Label(self.frame, text='Posterize Cutoff (%)').grid(row=3, column=0)
        self.posterize_cutoff_entry = ttk.Entry(self.frame)
        self.posterize_cutoff_entry.grid(row=3, column=2)
        self.posterize_button = ttk.Button(self.frame, text='NOT Posterizing')
        self.posterize_button.grid(row=3, column=3, sticky='ew')

        ttk.Label(self.frame, text='Fine Adj. Border (%)').grid(row=4, column=0)
        self.fine_adj_border_entry = ttk.Entry(self.frame)
        self.fine_adj_border_entry.grid(row=4, column=1, sticky='ew')
        self.fine_adj_button_1 = ttk.Button(self.frame, text='Button 1')
        self.fine_adj_button_1.grid(row=4, column=2, sticky='ew')
        self.fine_adj_button_2 = ttk.Button(self.frame, text='Button 2')
        self.fine_adj_button_2.grid(row=4, column=3, sticky='ew')

        for i, ty in enumerate(('Pattern', 'Red Focus', 'UV Focus')):
            ttk.Label(self.frame, text=f'{ty} Channels').grid(row=5+i, column=0)
            red = ToggleButton(self.frame, ButtonStyle('RED ENABLED', 'red'), ButtonStyle('Red Disabled', 'black'))
            red.widget.grid(row=5+i, column=1, sticky='ew')
            green = ToggleButton(self.frame, ButtonStyle('GREEN ENABLED', 'green'), ButtonStyle('Green Disabled', 'black'))
            green.widget.grid(row=5+i, column=2, sticky='ew')
            blue = ToggleButton(self.frame, ButtonStyle('BLUE ENABLED', 'blue'), ButtonStyle('Blue Disabled', 'black'))
            blue.widget.grid(row=5+i, column=3, sticky='ew')
        
        ttk.Label(self.frame, text='Calibration Controls').grid(row=8, column=0)
        self.calibrate_button = ttk.Button(self.frame, text='Calibrate')
        self.calibrate_button.grid(row=8, column=1, sticky='ew')
        self.calibrate_entry = ttk.Entry(self.frame)
        self.calibrate_entry.grid(row=8, column=2, sticky='ew')
        self.goto_button = ttk.Button(self.frame, text='goto DISABLED')
        self.goto_button.grid(row=8, column=3, sticky='ew')

def main():
    root = Tk()
    root.title('Lithography stepper')

    mainframe = ttk.Frame(root, padding='3 3 12 12')
    mainframe.grid(column=0, row=0, sticky='nesw')

    stage_notebook = ttk.Notebook(mainframe)
    stage_position_frame = StagePositionFrame(stage_notebook)
    fine_adjust_frame = ttk.Frame(stage_notebook)
    stage_notebook.add(stage_position_frame.frame, text='Stage Position')
    stage_notebook.add(fine_adjust_frame, text='Fine Adjust')


    options_notebook = ttk.Notebook(mainframe)
    options_frame = OptionsFrame(options_notebook)
    options_notebook.add(options_frame.frame, text='Options')

    
    stage_notebook.grid(row=0, column=1)
    options_notebook.grid(row=0, column=2)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    root.mainloop()

if __name__ == '__main__':
    main()