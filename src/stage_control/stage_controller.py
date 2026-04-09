# Hacker Fab
# J. Kent Wirant
# Stage Controller Interface


class UnsupportedCommand(Exception):
    pass


# This will be an abstract interface for stage positioning.
class StageController:

    def has_homing(self):
        return False
    
    def home(self):
        print(f"ignoring home() in dummy_stage controller")
    
    def move_relative(self, microns: dict[str, float]):
        print(f"ignoring move_relative {microns} in dummy_stage controller")

    def move_absolute(self, microns: dict[str, float]):
        print(f"ignoring move_absolute {microns} in dummy_stage controller")

    def soft_reset(self):
        print(f"ignoring soft_reset in dummy_stage controller")
    
    def set_on_start_location(self):
        print(f"ignoring set_on_start_location in dummy_stage controller")

    def get_autofocus(self):
        print(f"ignoring get_autofocus in dummy_stage controller")

    def get_position(self):
        print(f"ignoring get_position in dummy_stage controller")
    
    def get_on_start_location(self):
        print(f"ignoring get_on_start_location in dummy_stage controller")
    
    def get_bounds(self)->dict[str, tuple[float, float]]:
        print(f"ignoring get_bounds in dummy_stage controller")

