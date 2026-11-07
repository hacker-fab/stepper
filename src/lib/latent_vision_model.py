from pathlib import Path
import threading
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LATENT_VISION_CHECKPOINT = Path("checkpoints/latent_vision/best_model.pth")


def resolve_checkpoint_path(path: Union[str, Path] = DEFAULT_LATENT_VISION_CHECKPOINT) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    # Try finding relative to project root
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / p
    if candidate.is_file():
        return candidate
    return p


class RecursiveLevel(nn.Module):

    def __init__(self, in_channels: int, channels: int, out_channels: int, depth: int):
        super().__init__()
        self.depth = depth

        # Input goes through two conv2d to get x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.sub_level = None
        if depth > 1:
            # Smaller recursive level
            self.sub_level = RecursiveLevel(
                in_channels = channels,
                channels = channels * 2,
                out_channels = channels,
                depth=depth - 1,
            )
            post_in_channels = channels * 2
        else:
            post_in_channels = channels

        # Two conv layers before the final output on each level
        self.conv2 = nn.Sequential(
            nn.Conv2d(post_in_channels, channels, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)

        if self.sub_level is not None:
            x_down = F.max_pool2d(x1, kernel_size=2, stride=2)
            x_sub = self.sub_level(x_down)
            x2 = F.interpolate(x_sub, size=x1.shape[2:], mode="bilinear", align_corners=False)
            stacked = torch.cat([x1, x2], dim=1)
            out = self.conv2(stacked)
        else:
            out = self.conv2(x1)

        return out


class LatentVisionCNN(nn.Module):
    """
    Explicit convolutional neural network for same-size image-to-image mapping:
      - Input:  RGB image with shape (B, 3, H, W)
      - Output: Single-channel heatmap with shape (B, 1, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        latent_channels: int = 32,
        depth: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.depth = depth


        # Recursive multi-scale network
        self.net = RecursiveLevel(
            in_channels=in_channels,
            channels=self.latent_channels,
            out_channels=self.latent_channels,
            depth=self.depth,
        )

        # Final projection to desired output channels (same spatial dimensions)
        self.out_conv = nn.Conv2d(self.latent_channels, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.out_conv(feat)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))




def load_latent_vision_model(
    checkpoint_path: Union[str, Path], device: str = "cpu"
) -> Optional[LatentVisionCNN]:
    """Attempts to load the LatentVisionCNN model from a checkpoint file.

    Returns the loaded model in eval mode, or None if loading fails.
    """
    ckpt_file = resolve_checkpoint_path(checkpoint_path)
    if not ckpt_file.is_file():
        print(f"[Latent Vision] Checkpoint file not found: {ckpt_file}")
        return None

    try:
        model = LatentVisionCNN()
        ckpt = torch.load(ckpt_file, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            model = ckpt
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Latent Vision] Failed to load latent vision checkpoint: {e}")
        return None


class LatentVisionWorker(threading.Thread):
    """Asynchronous worker for latent vision model inference and mask upscaling.

    Runs in a background thread to prevent blocking camera updates and the GUI.
    Waits for at least min_frames (default 30) in the queue, averages them,
    and runs inference on the averaged frame.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        on_mask_ready: Callable[[np.ndarray], None],
        device: str = "cpu",
        min_frames: int = 30,
    ):
        super().__init__(daemon=True)
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.on_mask_ready = on_mask_ready
        self.device = device
        self.min_frames = min_frames

        self._frame_queue: list[np.ndarray] = []
        self._frame_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self.model: Optional[LatentVisionCNN] = None

    def submit_frame(self, frame: np.ndarray):
        """Submit a camera frame to the queue without blocking."""
        with self._frame_lock:
            self._frame_queue.append(frame)
            if len(self._frame_queue) > self.min_frames * 2:
                self._frame_queue.pop(0)
            queue_len = len(self._frame_queue)
        if queue_len >= self.min_frames:
            self._wake_event.set()

    def clear_queue(self):
        """Clear all pending frames from the queue."""
        with self._frame_lock:
            self._frame_queue.clear()

    def stop(self):
        """Signals the worker thread to stop."""
        self._stop_event.set()
        self._wake_event.set()

    def run(self):
        self.model = load_latent_vision_model(self.checkpoint_path, device=self.device)
        if self.model is None:
            return

        while not self._stop_event.is_set():
            self._wake_event.wait()
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            with self._frame_lock:
                if len(self._frame_queue) < self.min_frames:
                    continue
                frames_to_process = self._frame_queue
                self._frame_queue = []

            if not frames_to_process:
                continue

            try:
                # 1. Filter frames to ensure all match the latest frame shape
                latest_shape = frames_to_process[-1].shape
                matching_frames = [f for f in frames_to_process if f.shape == latest_shape]
                if len(matching_frames) < self.min_frames:
                    with self._frame_lock:
                        self._frame_queue = matching_frames + self._frame_queue
                    continue

                # 2. Average the frames
                averaged_frame = np.mean(matching_frames, axis=0, dtype=np.float32)
                frame_to_process = np.round(averaged_frame).astype(matching_frames[0].dtype)

                # 3. Ensure RGB format
                if frame_to_process.ndim == 2:
                    rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)

                h, w = rgb.shape[:2]

                # 4. Downscaled by x2 before feeding into the model
                down_w = max(1, w // 2)
                down_h = max(1, h // 2)
                downscaled = cv2.resize(rgb, (down_w, down_h), interpolation=cv2.INTER_AREA)

                # 5. Model accept RGB format, 0-1 float data
                img_float = downscaled.astype(np.float32) / 255.0

                # 6. Torch Tensor (1, 3, down_h, down_w)
                tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)

                # 7. Predict: output is same as input size
                out = self.model.predict(tensor)
                out_mask = out.squeeze().cpu().numpy()

                # 8. Scaled back up to match the camera image size
                scaled_mask = cv2.resize(out_mask, (w, h), interpolation=cv2.INTER_LINEAR)

                # 9. Notify callback with scaled mask
                self.on_mask_ready(scaled_mask)
            except Exception as e:
                print(f"[Latent Vision] Inference error: {e}")
