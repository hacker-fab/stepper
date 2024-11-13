# Hacker Fab
# J. Kent Wirant
# Stage Controller Interface

# This will be an abstract interface for stage positioning.
class StageController:
    def move_by(self, amounts: dict[str, float]):
        print(f'ignoring move_by {amounts} in dummy stage controller')

    def move_to(self, amounts: dict[str, float]):
        print(f'ignoring move_to {amounts} in dummy_stage controller')