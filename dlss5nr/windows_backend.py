# SPDX-License-Identifier: MIT
# Windows backend: in-process D3D12/NGX bridge via ctypes, following the
# MIT-licensed ComfyUI-DLSS5-NR integration (lisitskyaa/ComfyUI-DLSS5-NR).

from __future__ import annotations

import ctypes
import os
import platform
import threading
from pathlib import Path

import numpy as np

from . import common
from .common import DLSS5Error

_lock = threading.RLock()
# Serializes bridge calls into the shared in-process session (same pattern as
# the linux worker lock).
_worker_lock = threading.Lock()
_lib = None
_initialized_gpu: int | None = None
_dll_directory_handles: list = []


def supported() -> bool:
    return platform.system() == "Windows"


def _load_library(runtime: Path):
    global _lib
    if _lib is not None:
        return _lib
    bridge = common.PLUGIN_ROOT / "native" / "bin" / "dlss5nr_bridge.dll"
    if not bridge.exists():
        raise DLSS5Error(
            f"Native bridge is missing: {bridge}. Install the plugin completely "
            "(native/bin/dlss5nr_bridge.dll) or build it from native/src/build_native.bat."
        )
    for dll_dir in (bridge.parent, runtime, runtime / "caller"):
        if dll_dir.exists():
            _dll_directory_handles.append(os.add_dll_directory(str(dll_dir)))

    lib = ctypes.WinDLL(str(bridge))
    lib.dlss5nr_init.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_int]
    lib.dlss5nr_init.restype = ctypes.c_int
    # Signature mirrors dlss5nr_host.cpp ProcessFn (dlss5nr_process_v2).
    lib.dlss5nr_process_v2.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # input color (RGB float32, render size)
        ctypes.POINTER(ctypes.c_uint16),  # motion vectors (FP16 bits)
        ctypes.POINTER(ctypes.c_float),   # output color (RGB float32, output size)
        ctypes.c_int, ctypes.c_int,       # input w/h
        ctypes.c_int, ctypes.c_int,       # output w/h
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # style, preset, perf_quality
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,  # intensity/tone/structure/skin/global_tone
        ctypes.c_int, ctypes.c_int,       # automask, reset
        ctypes.c_char_p, ctypes.c_int,    # error buffer
    ]
    lib.dlss5nr_process_v2.restype = ctypes.c_int
    lib.dlss5nr_shutdown.argtypes = []
    lib.dlss5nr_shutdown.restype = None
    lib.dlss5nr_version.argtypes = []
    lib.dlss5nr_version.restype = ctypes.c_char_p
    lib.dlss5nr_gpu_name.argtypes = []
    lib.dlss5nr_gpu_name.restype = ctypes.c_char_p
    _lib = lib
    return lib


def _ensure_initialized(gpu_index: int, runtime: Path):
    lib = _load_library(runtime)
    global _initialized_gpu
    # The bridge reads the DLSSNR file name at first init from the process
    # environment; keep it in sync with the GPU actually being used.
    os.environ["DLSS5NR_SNR_FILENAME"] = common.resolve_snr_filename(runtime, gpu_index)
    if _initialized_gpu == gpu_index:
        return lib
    if _initialized_gpu is not None:
        lib.dlss5nr_shutdown()
        _initialized_gpu = None
    error = ctypes.create_string_buffer(4096)
    runtime_wstr = ctypes.c_wchar_p(str(runtime))
    if not lib.dlss5nr_init(int(gpu_index), runtime_wstr, error, len(error)):
        message = error.value.decode("utf-8", errors="replace")
        raise DLSS5Error(f"DLSS5 init failed: {message or 'unknown error'}")
    _initialized_gpu = gpu_index
    return lib


def _decode_error(buffer: ctypes.Array) -> str:
    return buffer.value.decode("utf-8", errors="replace")


def process_frames(frames, input_w: int, input_h: int, output_w: int, output_h: int,
                   params: dict, progress_callback=None):
    """Run frames through the in-process bridge.

    `frames` yields (rgb_u8, motion_fp16, reset); yields RGB float32 outputs.
    """
    common.validate_sizes(input_w, input_h, output_w, output_h)
    runtime: Path = params["runtime"]
    # The bridge reads this at carrier-feature creation (AutoExposure/IsHDR flags).
    os.environ["DLSS5NR_HDR"] = "1" if params.get("hdr") else "0"
    lib = _ensure_initialized(int(params["gpu_index"]), runtime)
    error = ctypes.create_string_buffer(4096)
    total = int(params["frame_count"])
    with _worker_lock:
        for index, (pixels_u8, motion, reset) in enumerate(frames):
            source = np.ascontiguousarray(pixels_u8, dtype=np.float32) / 255.0
            if params.get("hdr"):
                source = common.srgb_to_linear(source)
            motion_u16 = np.ascontiguousarray(motion, dtype=np.float16).view(np.uint16)
            output = np.zeros((output_h, output_w, 3), dtype=np.float32)
            ok = lib.dlss5nr_process_v2(
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                motion_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(input_w), int(input_h), int(output_w), int(output_h),
                int(params["style"]), int(params["preset"]), int(params["perf_quality"]),
                ctypes.c_float(float(params["intensity"])), ctypes.c_float(float(params["tone"])),
                ctypes.c_float(float(params["structure"])), ctypes.c_float(float(params["skin"])),
                ctypes.c_float(float(params["global_tone"])),
                1 if params["auto_mask"] else 0, 1 if reset else 0,
                error, len(error),
            )
            if not ok:
                raise DLSS5Error(f"DLSS5 frame {index} failed: {_decode_error(error) or 'unknown error'}")
            if params.get("detail", 1.0) != 1.0 or params.get("color", 1.0) != 1.0:
                output = common.nr_composite(
                    pixels_u8, output, params["detail"], params["color"],
                    out_is_linear=bool(params.get("hdr")), encode=not params.get("hdr"),
                )
            out = common.linear_to_srgb(output) if params.get("hdr") else output
            if progress_callback is not None:
                progress_callback(index + 1, total)
            yield out


def shutdown() -> None:
    global _initialized_gpu
    with _lock:
        if _lib is not None and _initialized_gpu is not None:
            try:
                _lib.dlss5nr_shutdown()
            except Exception:
                pass
        _initialized_gpu = None
