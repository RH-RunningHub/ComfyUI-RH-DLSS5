# SPDX-License-Identifier: MIT
# Temporal motion guides for DLSS5 NR, adapted from the MIT-licensed
# ComfyUI-DLSS5-NR-Linux frontend (kos94ok/ComfyUI-DLSS5-NR-Linux).

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # Optical flow is optional; zero motion remains a safe fallback.
    cv2 = None


def flow_implausible(flow: np.ndarray, basis: int) -> bool:
    """Heuristic bad-flow detector; True means "treat this frame as a cut".

    `flow` is (h, w, 2) pixels-per-frame at its native resolution and `basis`
    is the frame width at that resolution. Optical flow on fast pans, motion
    blur or repetitive texture can come back incoherent; feeding such vectors
    to the temporal NR pass accumulates ghosting, while a history reset only
    costs a single soft frame.
    """
    mag = np.hypot(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
    mean_mag = float(mag.mean())
    if mean_mag <= 1e-3:
        return False  # static frame: zero-ish flow is always safe
    if mean_mag > 0.35 * basis:
        return True  # implausibly large overall motion
    # Direction coherence: the median vector should carry a decent share of the
    # mean magnitude. Noise fields have a near-zero median.
    median = float(np.hypot(np.median(flow[..., 0]), np.median(flow[..., 1])))
    return median / mean_mag < 0.15


class TemporalGuideGenerator:
    """Generate Merserk-compatible FP16 motion guides from RGB frames.

    Frames are uint8 RGB arrays at render size. When OpenCV is unavailable the
    generator emits zero motion and a reset flag so temporal history is
    re-initialised per frame (safe, slightly less stable between frames).
    """

    def __init__(self, width: int, height: int, scene_change_threshold: float = 0.24) -> None:
        self.width = int(width)
        self.height = int(height)
        self.scene_change_threshold = float(scene_change_threshold)
        self.previous_gray: np.ndarray | None = None
        self.available = cv2 is not None
        if self.available:
            flow_width = min(self.width, 640)
            self.flow_width = max(64, int(round(flow_width / 2.0) * 2))
            self.flow_height = max(64, int(round(self.height * self.flow_width / self.width / 2.0) * 2))
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.dis.setUseSpatialPropagation(True)
            self.dis.setFinestScale(1)

    def _small_gray(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.flow_width, self.flow_height), interpolation=cv2.INTER_AREA)

    def process(self, rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.available:
            reset = self.previous_gray is None
            self.previous_gray = None
            return np.zeros((self.height, self.width, 2), dtype=np.float16), reset

        current = self._small_gray(rgb)
        if self.previous_gray is None:
            self.previous_gray = current
            return np.zeros((self.height, self.width, 2), dtype=np.float16), True

        scene_score = float(np.mean(cv2.absdiff(current, self.previous_gray))) / 255.0
        if scene_score > self.scene_change_threshold:
            motion = np.zeros((self.height, self.width, 2), dtype=np.float32)
            reset = True
        else:
            flow = self.dis.calc(current, self.previous_gray, None)
            if flow_implausible(flow, self.flow_width):
                motion = np.zeros((self.height, self.width, 2), dtype=np.float32)
                reset = True
            else:
                motion = cv2.resize(flow, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                motion[..., 0] *= self.width / self.flow_width
                motion[..., 1] *= self.height / self.flow_height
                reset = False
        self.previous_gray = current
        return np.ascontiguousarray(motion.astype(np.float16)), reset


def sample_u8(image_tensor: np.ndarray) -> np.ndarray:
    """float RGB [H,W,C] in 0..1 -> uint8 RGB for the guide generator."""
    return np.clip(image_tensor * 255.0 + 0.5, 0, 255).astype(np.uint8)
