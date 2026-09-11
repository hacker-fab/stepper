import os
import time
from typing import Callable, Optional

import cv2
import numpy as np
from core.operation import Operation, ExecutionContext


def compute_focus_score(camera_image: np.ndarray, blue_only: bool, ddepth=cv2.CV_64F, kernel_size=5, log: bool = False):
    if camera_image is None:
        return 0.0

    camera_image = camera_image.copy()
    camera_image[:, :, 1] = 0  # green should never be used for focus
    if blue_only:
        camera_image[:, :, 0] = 0  # disable red

    src = camera_image
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)

    # Apply Laplace function
    src = cv2.Laplacian(src, ddepth, ksize=kernel_size)

    return float(src.var())


def execute_autofocus(
    has_homing: bool,
    get_autofocus_base: Callable[[], float],
    get_current_z: Callable[[], float],
    move_absolute: Callable[[dict[str, float]], bool],
    move_relative: Callable[[dict[str, float]], bool],
    get_camera_image: Callable[[], Optional[np.ndarray]],
    delay: Callable[[float], None] = time.sleep,
    on_warning: Optional[Callable[[str], None]] = None,
    blue_only: bool = False,
    log: bool = False,
) -> bool:
    """Executes the autofocus routine independently of the GUI framework.

    Returns True if autofocus completed successfully, False otherwise.
    """
    if log:
        try:
            os.mkdir("aftest")
        except FileExistsError:
            pass
        log_file = open("aftest/log.csv", "w")
    else:
        log_file = None

    try:
        if has_homing:
            counter = 0

            def sample():
                nonlocal counter

                def one_sample():
                    img = get_camera_image()
                    return compute_focus_score(img, blue_only=blue_only, log=True)

                focus_score = sum([one_sample() for _ in range(5)]) / 5.0
                print("focus average:", focus_score)
                if log and log_file:
                    log_file.write(f"{counter},{focus_score}\n")
                    img = get_camera_image()
                    if img is not None:
                        cv2.imwrite(f"aftest/img{counter}.png", img)
                counter += 1
                return focus_score

            print("Starting Autofocus...")
            best_score = -1.0
            best_z = 0.0
            z_base = get_autofocus_base()

            # account for uv mode, where z-focus is different
            if blue_only:
                z_base += 50.0
                if not move_absolute({"z": z_base}):
                    if on_warning:
                        on_warning("Failed autofocus, z-stage can't go past boundary limits")
                    return False
                delay(1.0)
            else:
                for i in range(-20, 20, 2):
                    if not move_absolute({"z": z_base + i}):
                        if on_warning:
                            on_warning("Failed autofocus, z-stage can't go past boundary limits")
                        return False
                    delay(0.5)
                    new_score = sample()
                    if new_score > best_score:
                        best_score = new_score
                        best_z = get_current_z()

                print(f"Fine grain sampling done, best focus is: {best_score}")
                move_absolute({"z": best_z})
                delay(1.0)

        else:
            counter = 0

            def sample_focus():
                nonlocal counter

                def do_thing():
                    delay(0.1)
                    img = get_camera_image()
                    return compute_focus_score(img, blue_only=blue_only)

                focus_score = sorted([do_thing() for _ in range(3)])[1]
                if log and log_file:
                    log_file.write(f"{counter},{focus_score}\n")
                    img = get_camera_image()
                    if img is not None:
                        cv2.imwrite(f"aftest/img{counter}.png", img)
                counter += 1
                return focus_score

            delay(1.0)
            mid_score = sample_focus()
            move_relative({"z": -20.0})
            delay(1.0)
            neg_score = sample_focus()
            move_relative({"z": 40.0})
            delay(1.0)
            pos_score = sample_focus()
            move_relative({"z": -20.0})
            delay(1.0)

            last_focus = mid_score

            if neg_score < mid_score < pos_score:
                # Improved focus is in the +Z direction
                for i in range(30):
                    move_relative({"z": 10.0})
                    delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful +Z coarse autofocus {i}")
                        last_focus = new_score
                        break
                    last_focus = new_score

                for i in range(10):
                    move_relative({"z": -2.0})
                    delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful -Z fine autofocus {i}")
                        break
                    last_focus = new_score
            elif neg_score > mid_score > pos_score:
                # Improved focus is in the -Z direction
                for i in range(30):
                    move_relative({"z": -10.0})
                    delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful -Z coarse autofocus {i}")
                        break
                    last_focus = new_score

                for i in range(10):
                    move_relative({"z": 2.0})
                    delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful +Z fine autofocus {i}")
                        break
                    last_focus = new_score
            elif neg_score < mid_score and pos_score < mid_score:
                # We are very close to already being in focus
                print(f"Almost in focus! (neg {neg_score} mid {mid_score} pos {pos_score})")
                move_relative({"z": -20.0})
                delay(0.5)

                for i in range(30):
                    move_relative({"z": 2.0})
                    delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful +Z fine autofocus {i}")
                        break
                    last_focus = new_score
            else:
                print("Autofocus is confused!")

        print("Autofocus Complete.")
        return True

    finally:
        if log_file:
            log_file.close()


class AutofocusOperation(Operation):
    """Performs autofocus calibration."""

    def __init__(self, blue_only: bool = False, log: bool = False):
        super().__init__("Autofocus")
        self.blue_only = blue_only
        self.log = log

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]):
        report_progress(0.2, "Executing autofocus...")

        execute_autofocus(
            has_homing=context.stage.has_homing(),
            get_autofocus_base=lambda: context.stage.get_position()[2],
            get_current_z=lambda: context.stage.get_position()[2],
            move_absolute=context.stage.move_absolute,
            move_relative=context.stage.move_relative,
            get_camera_image=context.camera.get_latest_frame,
            delay=context.delay_func,
            on_warning=context.warning_callback,
            blue_only=self.blue_only,
            log=self.log,
        )
        report_progress(1.0, "Autofocus complete")