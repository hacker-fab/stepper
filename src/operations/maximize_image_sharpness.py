import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

from core.events import ColorMode
from core.operation import ExecutionContext, Operation
from lib.ar_tag import compute_focus_score


@dataclass
class SharpnessOptimizationResult:
    """Outcome of the sharpness maximization search."""
    best_z: float
    best_score: float
    iterations_completed: int
    converged: bool
    measured_points: Dict[float, float] = field(default_factory=dict)


class MaximizeImageSharpnessOperation(Operation):
    """Operation that maximizes image sharpness along the Z-axis without modifying the projector.

    Uses the Golden Section / Fibonacci Search method. Given the endpoints A and B,
    it samples x1 and x2 dividing the range into three segments, updating the search interval
    based on comparing sharpness scores.
    """

    def __init__(
        self,
        z_range: Union[float, Tuple[float, float]] = 20.0,
        threshold: float = 0.5,
        max_iterations: int = 10,
        settle_delay: float = 1,
        max_resamples: int = 3,
    ):
        super().__init__("Maximize Image Sharpness")
        self.z_range = z_range
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.settle_delay = settle_delay
        self.max_resamples = int(max_resamples)
        self.result: Optional[SharpnessOptimizationResult] = None

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.is_aborted:
            return "Sharpness maximization aborted"

        report_progress(0.05, "Starting image sharpness maximization...")

        current_z = context.stage.get_position()[2]

        # Determine initial search interval [z_low, z_high]
        if isinstance(self.z_range, (int, float)):
            half_range = abs(float(self.z_range))
            z_low = current_z - half_range
            z_high = current_z + half_range
        elif isinstance(self.z_range, (tuple, list)) and len(self.z_range) == 2:
            z_low = min(float(self.z_range[0]), float(self.z_range[1]))
            z_high = max(float(self.z_range[0]), float(self.z_range[1]))
        else:
            return f"Invalid z_range: {self.z_range}. Must be float or Tuple[float, float]."

        if z_high <= z_low:
            z_high = z_low + self.threshold

        # Cache for measured points: round(z, 4) -> score
        cache: Dict[float, float] = {}
        best_z = current_z
        best_score = -1.0

        def measure(z_target: float, force: bool = False) -> Optional[float]:
            nonlocal best_z, best_score
            key = round(float(z_target), 4)
            if not force and key in cache:
                return cache[key]

            if self.is_aborted:
                return None

            if not context.stage.move_absolute({"z": z_target}):
                msg = f"Failed to move Z stage to {z_target:.3f} µm (boundary limit reached)"
                if context.warning_callback:
                    context.warning_callback(msg)
                return None

            context.delay_func(self.settle_delay)
            frame = context.camera.get_latest_frame()
            score = compute_focus_score(frame)
            cache[key] = score

            if cache:
                best_k = max(cache, key=lambda k: cache[k])
                best_score = cache[best_k]
                best_z = best_k

            return score

        # Golden ratio conjugate constant: (sqrt(5) - 1) / 2
        inv_phi = (5.0**0.5 - 1.0) / 2.0

        a = z_low
        b = z_high

        # Sample x1 and x2 between a and b, dividing the range into three segments
        x1 = a + (1.0 - inv_phi) * (b - a)
        x2 = a + inv_phi * (b - a)

        s1 = measure(x1)
        if s1 is None:
            return "Sharpness maximization aborted" if self.is_aborted else f"Failed to move Z stage to {x1:.3f} µm"

        s2 = measure(x2)
        if s2 is None:
            return "Sharpness maximization aborted" if self.is_aborted else f"Failed to move Z stage to {x2:.3f} µm"

        iterations = 0
        converged = False

        while iterations < self.max_iterations:
            if (b - a) <= self.threshold:
                converged = True
                break

            if self.is_aborted:
                return "Sharpness maximization aborted"

            iterations += 1

            if s1 < s2:
                # The maximum point cannot be between a and x1, so set a to x1
                a = x1
                x1 = x2
                s1 = s2
                x2 = a + inv_phi * (b - a)
                s2 = measure(x2)
                if s2 is None:
                    return "Sharpness maximization aborted" if self.is_aborted else f"Failed to move Z stage to {x2:.3f} µm"
            else:
                # The maximum point cannot be between x2 and b, so set b to x2
                b = x2
                x2 = x1
                s2 = s1
                x1 = a + (1.0 - inv_phi) * (b - a)
                s1 = measure(x1)
                if s1 is None:
                    return "Sharpness maximization aborted" if self.is_aborted else f"Failed to move Z stage to {x1:.3f} µm"

            progress = min(1.0, iterations / max(1, self.max_iterations))
            report_progress(
                progress,
                f"Iter {iterations}/{self.max_iterations}: Z range [{a:.2f}, {b:.2f}], best score {best_score:.2f}",
            )

        if (b - a) <= self.threshold:
            converged = True

        # Finally, reposition stage to the global best Z position found
        if not context.stage.move_absolute({"z": best_z}):
            msg = f"Failed to return Z stage to best position {best_z:.3f} µm"
            if context.warning_callback:
                context.warning_callback(msg)
            return msg

        context.delay_func(self.settle_delay)

        self.result = SharpnessOptimizationResult(
            best_z=best_z,
            best_score=best_score,
            iterations_completed=iterations,
            converged=converged,
            measured_points=cache,
        )

        report_progress(1.0, f"Sharpness maximized at Z = {best_z:.2f} µm (score = {best_score:.2f})")
        return None
