# Hacker Fab
# J. Kent Wirant
# Stage Controller Interface


class UnsupportedCommand(Exception):
    pass


# This will be an abstract interface for stage positioning.
class StageController:
    def move_by(self, amounts: dict[str, float]):
        print(f"ignoring move_by {amounts} in dummy stage controller")

    def move_to(self, amounts: dict[str, float]):
        print(f"ignoring move_to {amounts} in dummy_stage controller")

    def has_homing(self):
        return False
    
    def home(self):
        print(f"ignoring home() in dummy_stage controller")
    
    def move_relative(self, microns: dict[str, float]):
        print(f"ignoring move_relative {microns} in dummy_stage controller")

    def move_absolute(self, microns: dict[str, float]):
        print(f"ignoring move_absolute {microns} in dummy_stage controller")

    def reset_and_unlock(self):
        print(f"ignoring unlock in dummy_stage controller")

    def get_position(self):
        print(f"ignoring get_position in dummy_stage controller")

