# SPDX-License-Identifier: MIT
# ComfyUI-RH-DLSS5 shared helpers.

from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]

MAGIC_DNR2 = b"DNR2"
MAGIC_FRM2 = b"FRM2"
MAGIC_OUT1 = b"OUT1"
MAGIC_END1 = b"END1"

# Binary transport between the Python frontend and dlss5nr_host.exe.
# magic + 12 uint32 fields + five float controls (two-size contract).
HEADER = struct.Struct("<4sIIIIIIIIIIII5f")
FRAME_HEADER = struct.Struct("<4sII")
REPLY_HEADER = struct.Struct("<4sIII")

# Fixed DLSS quality modes: perf_quality value -> display name.
PERF_QUALITY_NAMES = {
    5: "1x (DLAA / native)",
    2: "1.5x (Quality)",
    1: "1.724x (Balanced)",
    0: "2x (Performance)",
    3: "3x (Ultra Performance)",
}
SCALE_TO_PERF_QUALITY = {1.0: 5, 1.5: 2, 1.724: 1, 2.0: 0, 3.0: 3}

STYLE_VALUES = {"default": 0, "natural": 1, "cinematic": 2}

MAX_DIM = 16384
MAX_LONG_EDGE = 7680
MAX_SHORT_EDGE = 4320
MAX_PIXELS = 1 << 28
MAX_FRAMES = 1_000_000


class DLSS5Error(RuntimeError):
    """Actionable DLSS5 failure surfaced to the ComfyUI UI."""


def srgb_to_linear(x):
    """sRGB EOTF, piecewise exact (matches the reference host implementation)."""
    import numpy as np

    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(x):
    """Inverse sRGB EOTF. Model output may dip below 0; pow needs a clamped domain."""
    import numpy as np

    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(
        np.float32
    )


_LUMA_WEIGHTS = None


def _resize_float(img, width: int, height: int):
    """Bilinear resize for float HWC arrays (cv2 when present, PIL fallback)."""
    import numpy as np

    if img.shape[1] == width and img.shape[0] == height:
        return img
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image

        im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        im = im.resize((width, height), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def nr_composite(orig_u8, out, detail: float, color: float, out_is_linear: bool, encode: bool):
    """Reference-style composite of the NR result over the original, in linear light.

    Mirrors the reference tool's composite shader: the original's chroma is kept
    and rescaled to the model's luminance, `color` blends chroma toward the
    model, then `detail` blends the whole change over the original (0 = keep
    the original, 1 = full result, up to 2 = amplify). `out` is the model
    output (linear when out_is_linear, else sRGB-encoded); returns linear when
    encode is False, sRGB-encoded 0..1 otherwise.
    """
    import numpy as np

    global _LUMA_WEIGHTS
    if _LUMA_WEIGHTS is None:
        _LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    orig = srgb_to_linear(np.asarray(orig_u8, dtype=np.float32) / 255.0)
    nr = out if out_is_linear else srgb_to_linear(np.clip(np.asarray(out, dtype=np.float32), 0.0, None))
    if orig.shape[:2] != nr.shape[:2]:
        # Above 1x the composite base is the bilinear-upscaled original
        # (the reference shader's "linear upscaled original").
        orig = _resize_float(orig, nr.shape[1], nr.shape[0])
    lo = orig @ _LUMA_WEIGHTS
    ln = nr @ _LUMA_WEIGHTS
    scale = ln / np.maximum(lo, 1e-4)
    loC = orig * scale[..., None]
    cC = loC + (nr - loC) * np.float32(color)
    res = orig + (cC - orig) * np.float32(detail)
    res = np.clip(res, 0.0, 1.0)
    return linear_to_srgb(res) if encode else res


def even(value: float) -> int:
    result = int(round(value))
    return result - (result % 2)


def resolve_scale_factor(upscaling_mode: str) -> float:
    for scale, name in ((1.0, "1x"), (1.5, "1.5x"), (1.724, "1.724x"), (2.0, "2x"), (3.0, "3x")):
        if upscaling_mode.startswith(name):
            return scale
    raise DLSS5Error(f"Unknown upscaling mode: {upscaling_mode!r}")


def target_size(width: int, height: int, scale_factor: float) -> tuple[int, int]:
    if scale_factor == 1.0:
        return width, height
    out_w, out_h = even(width * scale_factor), even(height * scale_factor)
    ratio_x, ratio_y = out_w / width, out_h / height
    if abs(ratio_x - ratio_y) > 0.02:
        raise DLSS5Error("Neural DLSS upscaling requires a uniform scale factor.")
    ratio = (ratio_x + ratio_y) * 0.5
    closest = min(SCALE_TO_PERF_QUALITY, key=lambda candidate: abs(candidate - ratio))
    if abs(closest - ratio) > 0.03:
        supported = ", ".join(f"{candidate:g}x" for candidate in SCALE_TO_PERF_QUALITY)
        raise DLSS5Error(f"Unsupported DLSSNR scale factor {ratio:.3f}; choose one of {supported}")
    return out_w, out_h


def perf_quality_for(output_w: int, output_h: int, source_w: int, source_h: int) -> int:
    ratio = ((output_w / source_w) + (output_h / source_h)) * 0.5
    closest = min(SCALE_TO_PERF_QUALITY, key=lambda candidate: abs(candidate - ratio))
    if abs(closest - ratio) > 0.03:
        raise DLSS5Error(f"Unsupported DLSSNR scale factor {ratio:.3f}")
    return SCALE_TO_PERF_QUALITY[closest]


def validate_sizes(input_w: int, input_h: int, output_w: int, output_h: int) -> None:
    for name, value in (("input width", input_w), ("input height", input_h),
                        ("output width", output_w), ("output height", output_h)):
        if not 0 < value <= MAX_DIM:
            raise DLSS5Error(f"{name} {value} is outside the supported 1..{MAX_DIM} range")
    if max(output_w, output_h) > MAX_LONG_EDGE or min(output_w, output_h) > MAX_SHORT_EDGE:
        raise DLSS5Error(f"Output {output_w}x{output_h} exceeds the DLSSNR {MAX_LONG_EDGE}x{MAX_SHORT_EDGE} envelope")
    if input_w * input_h > MAX_PIXELS or output_w * output_h > MAX_PIXELS:
        raise DLSS5Error("Frame stream is too large for a single DLSS5 session (max 2^28 pixels per surface)")


def _comfy_models_dir() -> Path:
    """ComfyUI models root (persists across custom_nodes re-syncs)."""
    try:
        import folder_paths  # available inside ComfyUI
        return Path(folder_paths.models_dir)
    except Exception:
        return PLUGIN_ROOT.parent.parent / "models"


# The universal 310.8 DLSSNR build ships as nvngx_dlssnr.dll and covers RTX 20
# through 50 (including Blackwell SM120).  Dedicated per-lineage builds may sit
# alongside as nvngx_dlssnr_rtx30.dll / nvngx_dlssnr_rtx40.dll (same size,
# different weights) so one models/dlss5 directory can serve every generation.
SNR_DLL_DEFAULT = "nvngx_dlssnr.dll"
SNR_DLL_RTX30 = "nvngx_dlssnr_rtx30.dll"
SNR_DLL_RTX40 = "nvngx_dlssnr_rtx40.dll"


def detect_gpu_capability(gpu_index: int = 0) -> tuple[int, int] | None:
    """(major, minor) CUDA compute capability of the target GPU, or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        index = max(0, int(gpu_index))
        if index >= torch.cuda.device_count():
            index = 0
        major, minor = torch.cuda.get_device_capability(index)
        return int(major), int(minor)
    except Exception:
        return None


def resolve_snr_filename(runtime: Path, gpu_index: int = 0) -> str:
    """Pick the DLSSNR runtime DLL file name to load from `runtime`.

    DLSS5NR_SNR_FILENAME overrides everything (plain file name only).  Without
    it, Ampere/Ada GPUs (capability major 8) prefer their lineage build: Ada
    RTX 40 (SM89) loads nvngx_dlssnr_rtx40.dll, Ampere RTX 30 (SM80/86) loads
    nvngx_dlssnr_rtx30.dll and falls back to the rtx40 build when that file is
    missing; every other generation loads the universal nvngx_dlssnr.dll.
    """
    override = os.environ.get("DLSS5NR_SNR_FILENAME", "").strip()
    if override and "/" not in override and "\\" not in override and ".." not in override:
        return override
    capability = detect_gpu_capability(gpu_index)
    if capability is not None and capability[0] == 8:
        if capability[1] == 9:  # Ada / RTX 40
            if (runtime / SNR_DLL_RTX40).is_file():
                return SNR_DLL_RTX40
        else:  # Ampere / RTX 30 lineage
            if (runtime / SNR_DLL_RTX30).is_file():
                return SNR_DLL_RTX30
            if (runtime / SNR_DLL_RTX40).is_file():
                return SNR_DLL_RTX40
    return SNR_DLL_DEFAULT


def _ensure_caller_shim(runtime: Path) -> None:
    """The project caller shim ships inside the plugin, but the host expects it
    in <runtime>/caller. When the runtime dir is models/dlss5 (which only holds
    the three NVIDIA DLLs), link the bundled shim in (copy as fallback)."""
    bundled = PLUGIN_ROOT / "runtime" / "caller"
    target = runtime / "caller"
    if target.exists() or target.is_symlink():
        return
    if not (bundled / "nvngx.dll_comfy.dll").is_file():
        return
    try:
        target.symlink_to(bundled, target_is_directory=True)
    except OSError:
        try:
            shutil.copytree(bundled, target)
        except OSError:
            pass  # check_runtime_files reports the missing shim with a clear message


def resolve_runtime_dir(runtime_dir: str = "") -> Path:
    """Runtime directory resolution: widget override -> env -> models/dlss5 -> bundled runtime/."""
    candidates = []
    if runtime_dir and str(runtime_dir).strip():
        candidates.append(Path(str(runtime_dir).strip()))
    env_dir = os.environ.get("DLSS5_RUNTIME_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(_comfy_models_dir() / "dlss5")
    candidates.append(PLUGIN_ROOT / "runtime")
    for candidate in candidates:
        if (candidate / "nvngx_dlssnr.dll").is_file():
            _ensure_caller_shim(candidate)
            return candidate
    listed = ", ".join(str(c) for c in candidates)
    raise DLSS5Error(
        "No DLSS5 runtime was found (looked for nvngx_dlssnr.dll in: "
        f"{listed}). Place your legally obtained nvngx_dlssnr.dll in models/dlss5 "
        "(recommended, survives custom_nodes updates) or one of the other listed "
        "folders. See runtime/README.txt."
    )


def check_runtime_files(runtime: Path, scale_factor: float) -> None:
    missing = []
    if scale_factor > 1.0 and not (runtime / "nvngx_dlss.dll").is_file():
        missing.append(
            f"{runtime / 'nvngx_dlss.dll'} (required for upscaling above 1x; "
            "download it from the official NVIDIA DLSS SDK releases)"
        )
    if not (runtime / "caller" / "nvngx.dll_comfy.dll").is_file() and not (runtime / "caller" / "nvngx.dll").is_file():
        missing.append(f"{runtime / 'caller' / 'nvngx.dll_comfy.dll'} (project caller shim)")
    if missing:
        raise DLSS5Error("The DLSS5 runtime in "
                         f"{runtime} is incomplete. Missing: " + "; ".join(missing))


def style_int(style: str) -> int:
    try:
        return STYLE_VALUES[str(style).strip().lower()]
    except KeyError:
        raise DLSS5Error(f"Unknown DLSS5 style: {style!r}; expected one of {sorted(STYLE_VALUES)}")
