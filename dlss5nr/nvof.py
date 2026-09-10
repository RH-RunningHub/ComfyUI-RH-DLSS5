# SPDX-License-Identifier: MIT
# Hardware optical flow guides via the NVIDIA Optical Flow CUDA API (NVOFA).
# Same process() contract as TemporalGuideGenerator: (float16 (H,W,2) pixel
# displacement cur->prev, reset flag). Uses the driver's libnvidia-opticalflow
# through ctypes in torch's primary context; frame data moves through the
# CUDA runtime API (cudaMemcpy2DToArray/FromObject-free array copies) because
# torch's runtime initialisation disables the driver API's own copy/alloc
# entry points process-wide (CUDA_ERROR_INVALID_CONTEXT), while the NVOF
# library allocates its buffers internally via libnvcuvid and keeps working.
# Any setup or runtime failure degrades to zero motion + reset (the caller
# falls back to DIS optical flow at selection time).

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import platform
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_int32,
    c_size_t,
    c_uint32,
    c_uint64,
    c_void_p,
)

import numpy as np

from .motion import flow_implausible

# nvOpticalFlowCommon.h: NV_OF_API_MAJOR_VERSION 2, MINOR 0.
NV_OF_API_VERSION = (2 << 4) | 0

# cudaMemcpyKind
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2

_USAGE_INPUT, _USAGE_OUTPUT, _USAGE_COST = 1, 2, 4
_FMT_ABGR8, _FMT_SHORT2, _FMT_UINT8 = 3, 5, 7
_BUFFER_TYPE_CUARRAY = 1
_MODE_OPTICALFLOW = 1
_PERF_LEVELS = {"slow": 5, "medium": 10, "fast": 20}
_CAP_OUTPUT_GRIDS, _CAP_WIDTH_MIN, _CAP_HEIGHT_MIN, _CAP_WIDTH_MAX, _CAP_HEIGHT_MAX = 0, 4, 5, 6, 7


class NvofError(RuntimeError):
    pass


def _check(status: int, what: str) -> None:
    if status != 0:
        raise NvofError(f"{what} failed with NV_OF_STATUS {status}")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _even(value: int) -> int:
    return max(2, int(round(value / 2.0)) * 2)


# --- driver ABI -------------------------------------------------------------

_PFN_CREATE_OF = CFUNCTYPE(c_int, c_void_p, POINTER(c_void_p))
_PFN_OF_INIT = CFUNCTYPE(c_int, c_void_p, c_void_p)
_PFN_CREATE_BUFFER = CFUNCTYPE(c_int, c_void_p, c_void_p, c_int, POINTER(c_void_p))
_PFN_GET_CUARRAY = CFUNCTYPE(c_void_p, c_void_p)
_PFN_GET_CUDEVICEPTR = CFUNCTYPE(c_uint64, c_void_p)
_PFN_GET_STRIDE = CFUNCTYPE(c_int, c_void_p, c_void_p)
_PFN_SET_STREAMS = CFUNCTYPE(c_int, c_void_p, c_uint64, c_uint64)
_PFN_EXECUTE = CFUNCTYPE(c_int, c_void_p, c_void_p, c_void_p)
_PFN_DESTROY_BUFFER = CFUNCTYPE(c_int, c_void_p)
_PFN_DESTROY = CFUNCTYPE(c_int, c_void_p)
_PFN_LAST_ERROR = CFUNCTYPE(c_char_p, c_void_p)
_PFN_GET_CAPS = CFUNCTYPE(c_int, c_void_p, c_int, POINTER(c_uint32), POINTER(c_uint32))


class NV_OF_CUDA_API_FUNCTION_LIST(Structure):
    # nvOpticalFlowCuda.h _NV_OF_CUDA_API_FUNCTION_LIST (12 pointers, exact order).
    _fields_ = [
        ("nvCreateOpticalFlowCuda", _PFN_CREATE_OF),
        ("nvOFInit", _PFN_OF_INIT),
        ("nvOFCreateGPUBufferCuda", _PFN_CREATE_BUFFER),
        ("nvOFGPUBufferGetCUarray", _PFN_GET_CUARRAY),
        ("nvOFGPUBufferGetCUdeviceptr", _PFN_GET_CUDEVICEPTR),
        ("nvOFGPUBufferGetStrideInfo", _PFN_GET_STRIDE),
        ("nvOFSetIOCudaStreams", _PFN_SET_STREAMS),
        ("nvOFExecute", _PFN_EXECUTE),
        ("nvOFDestroyGPUBufferCuda", _PFN_DESTROY_BUFFER),
        ("nvOFDestroy", _PFN_DESTROY),
        ("nvOFGetLastError", _PFN_LAST_ERROR),
        ("nvOFGetCaps", _PFN_GET_CAPS),
    ]


class NV_OF_BUFFER_DESCRIPTOR(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("bufferUsage", c_int32),
        ("bufferFormat", c_int32),
    ]


class NV_OF_INIT_PARAMS(Structure):
    # uint32 width/height + six enum/int32 + pointer + two int32 = 48 bytes
    # (ctypes reproduces the C natural alignment incl. pointer padding).
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("outGridSize", c_int32),
        ("hintGridSize", c_int32),
        ("mode", c_int32),
        ("perfLevel", c_int32),
        ("enableExternalHints", c_int32),
        ("enableOutputCost", c_int32),
        ("hPrivData", c_void_p),
        ("disparityRange", c_int32),
        ("enableRoi", c_int32),
    ]


class NV_OF_EXECUTE_INPUT_PARAMS(Structure):
    _fields_ = [
        ("inputFrame", c_void_p),
        ("referenceFrame", c_void_p),
        ("externalHints", c_void_p),
        ("disableTemporalHints", c_int32),
        ("padding", c_uint32),
        ("hPrivData", c_void_p),
        ("padding2", c_uint32),
        ("numRois", c_uint32),
        ("roiData", c_void_p),
    ]


class NV_OF_EXECUTE_OUTPUT_PARAMS(Structure):
    _fields_ = [
        ("outputBuffer", c_void_p),
        ("outputCostBuffer", c_void_p),
        ("hPrivData", c_void_p),
    ]


class _STRIDE(Structure):
    _fields_ = [("strideXInBytes", c_uint32), ("strideYInBytes", c_uint32)]


class NV_OF_CUDA_BUFFER_STRIDE_INFO(Structure):
    _fields_ = [("strideInfo", _STRIDE * 3), ("numPlanes", c_uint32)]


def _load_nvof_library() -> ctypes.CDLL:
    errors = []
    names = ("nvofapi64.dll",) if platform.system() == "Windows" else ("libnvidia-opticalflow.so.1",)
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            errors.append(str(exc))
    for pattern in (
        "/usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so.*",
        "/usr/lib64/libnvidia-opticalflow.so.*",
    ):
        for path in sorted(glob.glob(pattern), reverse=True):
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                errors.append(str(exc))
    raise NvofError("libnvidia-opticalflow not loadable: " + "; ".join(errors[-2:]))


def _load_cudart() -> ctypes.CDLL:
    """Shared CUDA runtime, used only for the array copy entry points.

    torch's static runtime cannot be reached from ctypes, and the driver API's
    copy paths are disabled after torch initialises; a separate shared libcudart
    bound to the same primary context is the working route.
    """
    errors = []
    names = (("cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll")
             if platform.system() == "Windows"
             else ("libcudart.so.12", "libcudart.so.11.0", "libcudart.so"))
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            errors.append(str(exc))
    found = ctypes.util.find_library("cudart")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError as exc:
            errors.append(str(exc))
    for pattern in (
        "/usr/local/cuda*/lib64/libcudart.so*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so*",
    ):
        for path in sorted(glob.glob(pattern), reverse=True):
            if path.endswith(".a"):
                continue
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                errors.append(str(exc))
    raise NvofError("shared libcudart not loadable: " + "; ".join(errors[-2:]))


def _resize_bilinear(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if src.shape[0] == out_h and src.shape[1] == out_w:
        return src
    try:
        import cv2

        return cv2.resize(src, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        pass
    h, w = src.shape[:2]
    ys = np.linspace(0, h - 1, out_h)
    xs = np.linspace(0, w - 1, out_w)
    y0 = np.floor(ys).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    a = src[np.ix_(y0, x0)]
    b = src[np.ix_(y0, x1)]
    c = src[np.ix_(y1, x0)]
    d = src[np.ix_(y1, x1)]
    return (a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx).astype(
        src.dtype
    )


class _NvofBuffer:
    def __init__(self, handle: c_void_p, cuarray: c_void_p, row_bytes: int, height: int):
        self.handle = handle
        self.cuarray = cuarray
        self.row_bytes = row_bytes
        self.height = height


class NvofGuideGenerator:
    """NVOFA hardware optical flow generator, drop-in for TemporalGuideGenerator.

    Breaks (driver missing, no HW engine, mid-run error) degrade to zero motion
    + reset, never raising into the frame loop.
    """

    def __init__(self, width: int, height: int, scene_change_threshold: float = 0.24) -> None:
        self.width = int(width)
        self.height = int(height)
        self.scene_change_threshold = float(scene_change_threshold)
        self.available = False
        self.init_error = ""
        self.engine = "nvof"
        self._previous_gray: np.ndarray | None = None
        self._broken = False
        self._cost_enabled = False
        self._rt = None
        self._fn: NV_OF_CUDA_API_FUNCTION_LIST | None = None
        self._hof: c_void_p | None = None
        self._buffers: list[_NvofBuffer] = []
        try:
            self._engine_init()
            self.available = True
        except Exception as exc:
            self.init_error = f"{type(exc).__name__}: {exc}"
            self._release()

    # -- setup ---------------------------------------------------------------

    def _engine_init(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise NvofError("CUDA is not available")
        torch.zeros(8, device="cuda")
        torch.cuda.synchronize()

        self._rt = _load_cudart()
        self._rt.cudaSetDevice.argtypes = [c_int]
        self._rt.cudaSetDevice.restype = c_int
        self._rt.cudaDeviceSynchronize.argtypes = []
        self._rt.cudaDeviceSynchronize.restype = c_int
        self._rt.cudaMemcpy2DToArray.argtypes = [
            c_void_p, c_size_t, c_size_t, c_void_p, c_size_t, c_size_t, c_size_t, c_int,
        ]
        self._rt.cudaMemcpy2DToArray.restype = c_int
        self._rt.cudaMemcpy2DFromArray.argtypes = [
            c_void_p, c_size_t, c_void_p, c_size_t, c_size_t, c_size_t, c_size_t, c_int,
        ]
        self._rt.cudaMemcpy2DFromArray.restype = c_int
        _check(self._rt.cudaSetDevice(torch.cuda.current_device()), "cudaSetDevice")

        # Only a query is needed from the driver handle: the NVOF library wants
        # the context handle and does its own resource work through libnvcuvid.
        cuda = ctypes.CDLL("nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1")
        cuda.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        cuda.cuCtxGetCurrent.restype = c_int
        ctx = c_void_p()
        _check(cuda.cuCtxGetCurrent(byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            raise NvofError("no current CUDA context")

        lib = _load_nvof_library()
        fn = NV_OF_CUDA_API_FUNCTION_LIST()
        _check(
            lib.NvOFAPICreateInstanceCuda(c_uint32(NV_OF_API_VERSION), byref(fn)),
            "NvOFAPICreateInstanceCuda",
        )
        if not fn.nvCreateOpticalFlowCuda:
            raise NvofError("NvOF function list not populated")
        self._fn = fn
        self._ctx = ctx

        flow_width = _even(min(self.width, _env_int("DLSS5NR_OF_MAXW", 640)))
        flow_height = _even(int(round(self.height * flow_width / max(1, self.width))))
        grid = _env_int("DLSS5NR_OF_GRID", 2)
        perf = _env_int("DLSS5NR_OF_PERF", 10)
        cost_lo = _env_int("DLSS5NR_OF_COST_LO", 64)
        cost_hi = _env_int("DLSS5NR_OF_COST_HI", 192)
        self._cost_lo, self._cost_hi = float(min(cost_lo, cost_hi)), float(max(cost_lo, cost_hi))

        last_error: Exception | None = None
        for enable_cost in (True, False):
            try:
                self._init_session(ctx, flow_width, flow_height, grid, perf, enable_cost)
                self._cost_enabled = enable_cost
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                self._release_session()
                if not enable_cost:
                    raise
        if last_error is not None:  # pragma: no cover
            raise last_error

    def _init_session(
        self,
        ctx: c_void_p,
        flow_width: int,
        flow_height: int,
        grid: int,
        perf: int,
        enable_cost: bool,
    ) -> None:
        fn = self._fn
        hof = c_void_p()
        _check(fn.nvCreateOpticalFlowCuda(self._ctx, byref(hof)), "nvCreateOpticalFlowCuda")
        self._hof = hof

        grids = self._caps(_CAP_OUTPUT_GRIDS)
        if grids and grid not in grids:
            grid = min(grids)

        width_min = self._caps(_CAP_WIDTH_MIN)
        height_min = self._caps(_CAP_HEIGHT_MIN)
        flow_width = max(flow_width, width_min[0] if width_min else 0)
        flow_height = max(flow_height, height_min[0] if height_min else 0)

        init = NV_OF_INIT_PARAMS(
            width=flow_width,
            height=flow_height,
            outGridSize=grid,
            hintGridSize=grid,
            mode=_MODE_OPTICALFLOW,
            perfLevel=perf,
            enableExternalHints=0,
            enableOutputCost=1 if enable_cost else 0,
            hPrivData=None,
            disparityRange=0,
            enableRoi=0,
        )
        _check(fn.nvOFInit(self._hof, byref(init)), "nvOFInit")

        self.flow_width, self.flow_height, self.grid = flow_width, flow_height, grid
        self._out_w, self._out_h = -(-flow_width // grid), -(-flow_height // grid)
        self._buf_prev = self._create_buffer(flow_width, flow_height, _USAGE_INPUT, _FMT_ABGR8)
        self._buf_cur = self._create_buffer(flow_width, flow_height, _USAGE_INPUT, _FMT_ABGR8)
        self._buf_out = self._create_buffer(
            self._out_w, self._out_h, _USAGE_OUTPUT, _FMT_SHORT2
        )
        self._buf_cost = (
            self._create_buffer(self._out_w, self._out_h, _USAGE_COST, _FMT_UINT8)
            if enable_cost
            else None
        )

    def _caps(self, cap: int, count: int = 16) -> list[int]:
        vals = (c_uint32 * count)()
        returned = c_uint32(count)
        _check(self._fn.nvOFGetCaps(self._hof, cap, vals, byref(returned)), f"nvOFGetCaps({cap})")
        return [int(vals[i]) for i in range(min(returned.value, count))]

    def _create_buffer(self, width: int, height: int, usage: int, fmt: int) -> _NvofBuffer:
        desc = NV_OF_BUFFER_DESCRIPTOR(width, height, usage, fmt)
        handle = c_void_p()
        _check(
            self._fn.nvOFCreateGPUBufferCuda(
                self._hof, byref(desc), _BUFFER_TYPE_CUARRAY, byref(handle)
            ),
            "nvOFCreateGPUBufferCuda",
        )
        stride = NV_OF_CUDA_BUFFER_STRIDE_INFO()
        _check(
            self._fn.nvOFGPUBufferGetStrideInfo(handle, byref(stride)),
            "nvOFGPUBufferGetStrideInfo",
        )
        cuarray = c_void_p(self._fn.nvOFGPUBufferGetCUarray(handle))
        if not cuarray.value:
            self._fn.nvOFDestroyGPUBufferCuda(handle)
            raise NvofError("NVOF buffer has no CUarray")
        buf = _NvofBuffer(handle, cuarray, int(stride.strideInfo[0].strideXInBytes), height)
        self._buffers.append(buf)
        return buf

    # -- teardown ------------------------------------------------------------

    def _release_session(self) -> None:
        fn = self._fn
        for buf in self._buffers:
            try:
                fn.nvOFDestroyGPUBufferCuda(buf.handle)
            except Exception:
                pass
        self._buffers = []
        if self._hof is not None and fn is not None:
            try:
                fn.nvOFDestroy(self._hof)
            except Exception:
                pass
        self._hof = None

    def _release(self) -> None:
        self._release_session()
        self._rt = None
        self._fn = None

    def __del__(self) -> None:
        try:
            self._release()
        except Exception:
            pass

    # -- per-frame -----------------------------------------------------------

    def _upload(self, buf: _NvofBuffer, bgra: np.ndarray) -> None:
        row_bytes = bgra.shape[1] * 4
        if buf.row_bytes < row_bytes:
            raise NvofError(f"buffer pitch {buf.row_bytes} < row {row_bytes}")
        _check(
            self._rt.cudaMemcpy2DToArray(
                buf.cuarray, 0, 0, c_void_p(bgra.ctypes.data), row_bytes, row_bytes, buf.height,
                _CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy2DToArray",
        )

    def _download(self, buf: _NvofBuffer, host: np.ndarray) -> None:
        row_bytes = host.nbytes // host.shape[0]
        if buf.row_bytes < row_bytes:
            raise NvofError(f"buffer pitch {buf.row_bytes} < row {row_bytes}")
        _check(
            self._rt.cudaMemcpy2DFromArray(
                c_void_p(host.ctypes.data), row_bytes, buf.cuarray, 0, 0, row_bytes, host.shape[0],
                _CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy2DFromArray",
        )

    def _downscale(self, rgb: np.ndarray) -> np.ndarray:
        """Frames arrive at render size; the engine consumes flow-resolution frames."""
        h, w = rgb.shape[:2]
        if w == self.flow_width and h == self.flow_height:
            return rgb
        try:
            import cv2

            return cv2.resize(rgb, (self.flow_width, self.flow_height), interpolation=cv2.INTER_AREA)
        except Exception:
            return _resize_bilinear(rgb, self.flow_height, self.flow_width)

    def _pack_bgra(self, rgb: np.ndarray) -> np.ndarray:
        # NV_OF_BUFFER_FORMAT_ABGR8 == D3D12 B8G8R8A8_UNORM: bytes B,G,R,A.
        rgb = self._downscale(rgb)
        h, w = rgb.shape[:2]
        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[..., 0] = rgb[..., 2]
        bgra[..., 1] = rgb[..., 1]
        bgra[..., 2] = rgb[..., 0]
        bgra[..., 3] = 255
        return bgra

    def _scene_gray(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        ys = np.linspace(0, h - 1, min(h, 288), dtype=np.int64)
        xs = np.linspace(0, w - 1, min(w, 512), dtype=np.int64)
        small = rgb[np.ix_(ys, xs)].astype(np.float32)
        return small[..., 0] * 0.299 + small[..., 1] * 0.587 + small[..., 2] * 0.114

    def _zero(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 2), dtype=np.float16)

    def process(self, rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        if self._broken or not self.available:
            return self._zero(), True
        try:
            return self._process_impl(rgb)
        except Exception as exc:
            self._broken = True
            print(f"[RH-DLSS5] nvof engine disabled after error: {exc}", flush=True)
            return self._zero(), True

    def _process_impl(self, rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        gray = self._scene_gray(rgb)
        bgra = self._pack_bgra(rgb)
        if self._previous_gray is None:
            self._previous_gray = gray
            self._upload(self._buf_prev, bgra)
            return self._zero(), True

        scene_score = float(np.mean(np.abs(gray - self._previous_gray))) / 255.0
        self._previous_gray = gray
        self._upload(self._buf_cur, bgra)
        if scene_score > self.scene_change_threshold:
            self._buf_prev, self._buf_cur = self._buf_cur, self._buf_prev
            return self._zero(), True

        inp = NV_OF_EXECUTE_INPUT_PARAMS(
            inputFrame=self._buf_cur.handle,
            referenceFrame=self._buf_prev.handle,
            disableTemporalHints=0,
        )
        outp = NV_OF_EXECUTE_OUTPUT_PARAMS(
            outputBuffer=self._buf_out.handle,
            outputCostBuffer=self._buf_cost.handle if self._buf_cost is not None else None,
        )
        _check(self._fn.nvOFExecute(self._hof, byref(inp), byref(outp)), "nvOFExecute")
        _check(self._rt.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

        flow = np.empty((self._out_h, self._out_w, 2), dtype=np.int16)
        self._download(self._buf_out, flow)
        motion = flow.astype(np.float32) / 32.0  # S10.5 -> pixels at flow res
        if flow_implausible(motion, self.flow_width):
            self._buf_prev, self._buf_cur = self._buf_cur, self._buf_prev
            return self._zero(), True
        if self._buf_cost is not None:
            cost = np.empty((self._out_h, self._out_w), dtype=np.uint8)
            self._download(self._buf_cost, cost)
            weight = np.clip(
                (self._cost_hi - cost.astype(np.float32))
                / max(1.0, self._cost_hi - self._cost_lo),
                0.0,
                1.0,
            )
            motion *= weight[..., None]

        motion = _resize_bilinear(motion, self.height, self.width)
        motion[..., 0] *= self.width / self.flow_width
        motion[..., 1] *= self.height / self.flow_height
        self._buf_prev, self._buf_cur = self._buf_cur, self._buf_prev
        return np.ascontiguousarray(motion.astype(np.float16)), False
