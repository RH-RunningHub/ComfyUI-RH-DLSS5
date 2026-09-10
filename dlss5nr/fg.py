# SPDX-License-Identifier: MIT
# DLSS frame interpolation (NGX DLSS-FG) for ComfyUI-RH-DLSS5.
#
# Streams RGB frames to the platform's dlssg-worker.exe (a direct D3D12 NGX
# host) under Wine and interleaves the generated frames between the sources.
# The worker is a user-supplied runtime binary, like the NVIDIA DLLs: it is
# never redistributed with the plugin. Expected runtime layout (see
# runtime/README.txt):
#
#   <runtime>/dlssg-worker.exe      direct NGX DLSS-G host (MIT, Merserk lineage)
#   <runtime>/nvngx.dll             NVIDIA NGX SDK loader stub
#   <runtime>/_nvngx.dll            NVIDIA NGX driver core (same file as models/dlss5)
#   <runtime>/nvngx_dlssg.dll       NVIDIA DLSS frame generation snippet
#
# Wire protocol (verified on RTX 4090 / wine 11 / DXVK, 2026-09):
#   setup  "<5I"   0x31534746 width height frame_count generated_count
#   reply  "<4I"   0x31524746 status multi_frame_count_max reserved
#   frame  "<4I2q" 0x31464746 index reset 0 ts_num ts_den + RGBA8 + f16 motion
#   reply  "<4I"   0x314F4746 status generated disabled + generated * RGBA8
#
# Semantics (measured, synthetic moving block): evaluating input frame i emits
# the interpolated frame(s) for the interval (i-1, i) - e.g. the frame returned
# for input 1 sits at t = 0.5 / fps. The final interval (T-1, T) has no future
# frame to interpolate towards, so a 2x run yields 2T-1 output frames.

from __future__ import annotations

import json
import os
import platform
import struct
import subprocess
import threading
from contextlib import closing
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from .common import DLSS5Error
from .linux_backend import _start_xvfb_if_needed, find_wine

SETUP_MAGIC = 0x31534746
SETUP_OUT_MAGIC = 0x31524746
FRAME_MAGIC = 0x31464746
FRAME_OUT_MAGIC = 0x314F4746
REPLY = struct.calcsize("<4I")
SETUP = struct.calcsize("<5I")
FRAME_HEAD = struct.calcsize("<4I2q")

WORKER_NAME = "dlssg-worker.exe"
RUNTIME_FILES = ("dlssg-worker.exe", "nvngx.dll", "_nvngx.dll", "nvngx_dlssg.dll")
MULTIPLIERS = {"2x": 2, "3x": 3, "4x": 4}
OUTPUT_FPS_CHOICES = [*MULTIPLIERS, "23.976", "24", "25", "29.97", "30", "48", "50",
                      "59.94", "60", "72", "90", "96", "120", "144"]
_FPS_ALIASES = {"23.976": Fraction(24000, 1001), "29.97": Fraction(30000, 1001),
                "59.94": Fraction(60000, 1001)}


def as_fps(value) -> Fraction:
    """Keep rational metadata exact; map conventional decimal NTSC aliases."""
    if isinstance(value, Fraction):
        return value
    text = str(value).strip()
    return _FPS_ALIASES[text] if text in _FPS_ALIASES else Fraction(text)


def parse_output_fps(choice) -> Fraction:
    """Widget fps value -> exact target rate (multiplier choices never reach here)."""
    return as_fps(str(choice).strip())


def _plan(src_fps: Fraction, multiplier: int, target_fps: Fraction | None):
    """Decide the 2x-cascade depth and the final timeline.

    Returns (stages, out_fps, exact_multiplier). exact_multiplier is set only
    for plain multiplier runs whose grid IS the output timeline (2x / 4x);
    every other target resamples the densest grid via _pick_indices.
    """
    src_fps = as_fps(src_fps)
    if src_fps <= 0:
        raise DLSS5Error("Source frame rate must be positive.")
    if target_fps is not None:
        target_fps = as_fps(target_fps)
        if target_fps <= src_fps:
            raise DLSS5Error(
                f"target output fps ({float(target_fps):g}) must exceed the source "
                f"rate ({float(src_fps):g}); interpolation only increases frame rate.")
        stages = 0
        while src_fps * (2 ** stages) < target_fps:
            stages += 1
        if target_fps > src_fps * 6:
            raise DLSS5Error(
                f"target output fps ({float(target_fps):g}) exceeds 6x the source rate "
                f"({float(src_fps):g}) and is not supported.")
        return stages, target_fps, None
    if multiplier not in (2, 3, 4):
        raise DLSS5Error("Supported multipliers are 2x, 3x and 4x.")
    stages = {2: 1, 3: 2, 4: 2}[multiplier]
    return stages, src_fps * multiplier, multiplier


def find_fg_runtime(override: str = "") -> Path:
    """Runtime folder holding the DLSS-G worker and its NVIDIA snippets."""
    candidates = []
    if override and str(override).strip():
        candidates.append(Path(str(override).strip()))
    env_dir = os.environ.get("DLSS5_FG_RUNTIME_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    try:
        from .common import resolve_runtime_dir

        dlss5_dir = resolve_runtime_dir("")
        candidates.append(dlss5_dir / "dlssg")
    except DLSS5Error:
        pass
    from .common import PLUGIN_ROOT

    candidates.append(PLUGIN_ROOT / "runtime" / "dlssg")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in RUNTIME_FILES):
            return candidate
    listed = ", ".join(str(c) for c in candidates)
    raise DLSS5Error(
        "DLSS frame generation runtime is missing. Place dlssg-worker.exe, nvngx.dll, "
        f"_nvngx.dll and nvngx_dlssg.dll in one of: {listed} (or set DLSS5_FG_RUNTIME_DIR). "
        "The worker binary and NVIDIA DLLs are user-supplied and never redistributed."
    )


_PROBE_CACHE: dict[str, tuple[float, dict]] = {}


def probe_fg_capabilities(runtime: Path) -> dict:
    """Run the worker's --probe capability query (cached per directory+mtime)."""
    key = str(runtime)
    stamp = max((runtime / name).stat().st_mtime for name in RUNTIME_FILES)
    cached = _PROBE_CACHE.get(key)
    if cached and cached[0] == stamp:
        return cached[1]

    env = _worker_env(runtime)
    proc = subprocess.Popen(
        _worker_command(runtime, "--probe"),
        cwd=str(runtime), env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise DLSS5Error("DLSS frame generation capability probe timed out.")
    if proc.returncode:
        detail = (stderr or stdout).strip() or f"exit code {proc.returncode}"
        raise DLSS5Error(f"DLSS frame generation probe failed:\n{detail[-2000:]}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise DLSS5Error("DLSS frame generation probe returned no result.")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise DLSS5Error(f"DLSS frame generation probe returned invalid JSON: {exc}")
    _PROBE_CACHE[key] = (stamp, result)
    return result


class _StderrDrain(threading.Thread):
    def __init__(self, stream) -> None:
        super().__init__(daemon=True)
        self.lines: deque[str] = deque(maxlen=80)
        self._stream = stream

    def run(self) -> None:
        for raw in iter(self._stream.readline, b""):
            try:
                self.lines.append(raw.decode("utf-8", errors="replace").rstrip())
            except Exception:
                pass

    def tail(self, count: int = 12) -> str:
        return "\n".join(list(self.lines)[-count:])


def _worker_command(runtime: Path, *args: str) -> list[str]:
    """Worker launch line: the D3D12 host runs natively on Windows, via Wine elsewhere."""
    worker = str(runtime / WORKER_NAME)
    if platform.system() == "Windows":
        return [worker, *args]
    return [str(find_wine()), worker, *args]


def _worker_env(runtime: Path) -> dict:
    env = os.environ.copy()
    if platform.system() == "Windows":
        return env
    env.setdefault("WINEDEBUG", "-all")
    env.setdefault("DXVK_ENABLE_NVAPI", "1")
    env.setdefault("WINEDLLOVERRIDES", "d3d12,d3d12core,nvapi64,dxgi=n,b")
    env.setdefault("DXVK_LOG_LEVEL", "none")
    env.setdefault("VKD3D_DEBUG", "none")
    # DXVK cannot enumerate the NVIDIA adapter without a display; headless
    # machines get a private Xvfb (same policy as the NR wine host).
    _start_xvfb_if_needed(env)
    return env


class FGSession:
    """One dlssg-worker.exe evaluation session (fixed size and frame budget)."""

    def __init__(self, runtime: Path, width: int, height: int, frame_count: int,
                 generated_count: int, wine_prefix: str = "") -> None:
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * 4
        self.generated_count = int(generated_count)
        env = _worker_env(runtime)
        if wine_prefix:
            env["WINEPREFIX"] = str(wine_prefix)
        try:
            self.proc = subprocess.Popen(
                _worker_command(runtime, "--serve"),
                cwd=str(runtime), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise DLSS5Error(f"Could not start the DLSS frame generation worker: {exc}")
        self.drain = _StderrDrain(self.proc.stderr)
        self.drain.start()
        self._closed = False
        try:
            self.proc.stdin.write(struct.pack("<5I", SETUP_MAGIC, self.width, self.height,
                                              max(1, int(frame_count)), self.generated_count))
            self.proc.stdin.flush()
            magic, status, maximum, _reserved = struct.unpack("<4I", self._read(REPLY))
        except Exception as exc:
            self.close()
            raise DLSS5Error(f"DLSS frame generation handshake failed:\n{self.drain.tail()}\n{exc}")
        if magic != SETUP_OUT_MAGIC or status:
            self.close()
            raise DLSS5Error(
                f"DLSS frame generation session rejected (status {status}); runtime "
                f"maximum is {maximum + 1}x.\n{self.drain.tail()}"
            )
        if self.generated_count > maximum:
            self.close()
            raise DLSS5Error(
                f"DLSS frame generation runtime supports up to {maximum + 1}x "
                f"(MultiFrameCountMax {maximum}), requested {self.generated_count + 1}x."
            )

    def _read(self, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            block = self.proc.stdout.read(count - len(data))
            if not block:
                raise DLSS5Error(f"DLSS frame generation worker closed its output:\n{self.drain.tail()}")
            data.extend(block)
        return bytes(data)

    def process_frame(self, rgba: np.ndarray, motion: np.ndarray, timestamp: Fraction,
                      reset: bool, index: int) -> list[np.ndarray]:
        if self._closed:
            raise DLSS5Error("DLSS frame generation session is closed.")
        color = np.ascontiguousarray(rgba, dtype=np.uint8)
        vectors = np.ascontiguousarray(motion, dtype=np.float16)
        if color.shape != (self.height, self.width, 4):
            raise DLSS5Error(f"DLSS frame generation color frame has unexpected shape {color.shape}.")
        if vectors.shape != (self.height, self.width, 2):
            raise DLSS5Error(f"DLSS frame generation motion field has unexpected shape {vectors.shape}.")
        try:
            self.proc.stdin.write(struct.pack("<4I2q", FRAME_MAGIC, index, 1 if reset else 0, 0,
                                              int(timestamp.numerator), int(timestamp.denominator)))
            self.proc.stdin.write(memoryview(color).cast("B"))
            self.proc.stdin.write(memoryview(vectors).cast("B"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DLSS5Error(f"DLSS frame generation pipeline I/O failed on frame {index}:\n"
                             f"{self.drain.tail()}\n{exc}")
        magic, status, generated, disabled = struct.unpack("<4I", self._read(REPLY))
        if magic != FRAME_OUT_MAGIC or status:
            raise DLSS5Error(f"DLSS frame generation evaluation failed at input frame {index} "
                             f"(status {status}):\n{self.drain.tail()}")
        if disabled:
            return []
        return [np.frombuffer(self._read(self.frame_bytes), np.uint8)
                .reshape(self.height, self.width, 4).copy() for _ in range(generated)]

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        try:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        except OSError:
            pass


def _guide_generator(width: int, height: int, motion_mode: str, threshold: float):
    if motion_mode in ("auto", "nvof"):
        try:
            from .nvof import NvofGuideGenerator

            return NvofGuideGenerator(width, height, threshold), "nvof"
        except Exception:
            if motion_mode == "nvof":
                raise
    from .motion import TemporalGuideGenerator

    return TemporalGuideGenerator(width, height, threshold), "dis"


def _frame_to_rgba(frame: torch.Tensor) -> np.ndarray:
    """One input frame (H, W, 3) float 0..1 -> (H, W, 4) RGBA uint8."""
    rgb = frame.detach().cpu().numpy()
    rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, np.uint8)
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))


def _rgba_into(out: torch.Tensor, index: int, rgba: np.ndarray) -> None:
    out[index] = torch.from_numpy(rgba[..., :3].astype(np.float32) * (1.0 / 255.0))


def _image_source(images: torch.Tensor):
    """Source accessor yielding (rgba_u8, original_float_frame) per input frame."""
    def source(index: int):
        frame = images[index].detach().cpu().to(torch.float32)
        return _frame_to_rgba(frame), frame
    return source


def _rgb_source(rgb: np.ndarray):
    """Indexable (T, H, W, 3) uint8 buffer -> per-frame RGBA accessor."""
    def source(index: int):
        rgba = np.empty((rgb.shape[1], rgb.shape[2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb[index]
        rgba[..., 3] = 255
        return rgba, None
    return source


def _rgba_source(cur: np.ndarray):
    def source(index: int):
        return cur[index], None
    return source


def _stage_frames(sources, count: int, runtime: Path, stage_fps: Fraction, wine_prefix: str,
                  threshold: float, motion_mode: str, progress, engine_out=None):
    """Yield (rgba_u8, src_float_or_None) frames in temporal order for one 2x pass.

    For input k first the generated frame of interval (k-1, k) (when present),
    then the source frame k itself. `src_float` carries the untouched input
    tensor slice so 2x runs never round-trip their sources through uint8.
    The FG session lives for the whole generator and is closed when the
    generator is exhausted or closed. Only one frame is materialised at a time
    beyond whatever the consumer keeps.
    """
    first_rgba, first_float = sources(0)
    height, width = first_rgba.shape[:2]
    guide, engine = _guide_generator(width, height, motion_mode, threshold)
    if engine_out is not None:
        engine_out.append(engine)
    zero = np.zeros((height, width, 2), np.float16)
    previous = (first_rgba, first_float)
    session = FGSession(runtime, width, height, count, 1, wine_prefix)
    try:
        for index in range(count):
            if index == 0:
                rgba, src_float = first_rgba, first_float
            else:
                rgba, src_float = sources(index)
            motion, reset = guide.process(rgba[..., :3])
            if index == 0:
                reset = True
            if reset:
                motion = zero
            got = session.process_frame(rgba, motion, Fraction(index, 1) / stage_fps,
                                        reset, index)
            if index:
                # Every interval has one slot, even when FG resets/is disabled.
                # Hold the previous source across cuts; never blend scenes.
                yield (got[0], None) if got and not reset else previous
            yield rgba, src_float
            previous = (rgba, src_float)
            if progress is not None:
                progress(index + 1, count)
    finally:
        session.close()


def _pick_indices(grid_count: int, grid_rate: Fraction, target_rate: Fraction) -> list[int]:
    """Map exact target_rate timestamps onto the nearest denser CFR grid point.

    Exact half-grid ties alternate direction so a persistent early/late bias
    cannot accumulate (same policy as the reference scheduler).
    """
    count = int(Fraction(grid_count - 1, 1) / grid_rate * target_rate) + 1
    indices = []
    tie_late = False
    for k in range(count):
        pos = Fraction(k, 1) / target_rate * grid_rate
        lower = pos.numerator // pos.denominator
        remainder = pos - lower
        if remainder < Fraction(1, 2):
            index = lower
        elif remainder > Fraction(1, 2):
            index = lower + 1
        else:
            index = lower + int(tie_late)
            tie_late = not tie_late
        indices.append(max(0, min(index, grid_count - 1)))
    return indices


def _prepare(src_fps, width: int, height: int, frames_count: int, runtime_dir: str):
    """Shared validation + runtime/probe gate for both interpolation entry points."""
    if frames_count < 2:
        raise DLSS5Error("DLSS frame interpolation needs at least 2 frames.")
    runtime = find_fg_runtime(runtime_dir)
    probe = probe_fg_capabilities(runtime)
    if not probe.get("available"):
        raise DLSS5Error(f"DLSS frame generation is not available on this machine: "
                         f"{probe.get('detail') or probe}")
    src_fps = as_fps(src_fps)
    if src_fps <= 0:
        raise DLSS5Error(f"Invalid source frame rate: {src_fps}")
    if width % 2 or height % 2:
        raise DLSS5Error("DLSS frame generation needs even frame dimensions.")
    return runtime, src_fps


def _sampled_frames(sources, count, runtime, src_fps, stages, out_fps,
                    wine_prefix, threshold, motion_mode, progress_callback, engines):
    """Buffer only non-final RGBA grids; sample the last pass as it is emitted."""
    total = sum((count - 1) * 2 ** stage + 1 for stage in range(stages))
    done = 0

    def progress(current, _total):
        if progress_callback is not None:
            progress_callback(done + current, total)

    for stage in range(stages):
        grid_count = 2 * count - 1
        frames = _stage_frames(sources, count, runtime, src_fps * 2 ** stage,
                               wine_prefix, threshold, motion_mode, progress, engines)
        with closing(frames):
            if stage == stages - 1:
                picks = iter(_pick_indices(grid_count, src_fps * 2 ** stages, out_fps))
                wanted = next(picks, None)
                for index, item in enumerate(frames):
                    if index == wanted:
                        yield item
                        wanted = next(picks, None)
                # Exhaust the stage even when its last grid point isn't selected:
                # progress and worker cleanup must still finish.
            else:
                buffer = None
                for index, (rgba, _) in enumerate(frames):
                    if buffer is None:
                        buffer = np.empty((grid_count, *rgba.shape), dtype=np.uint8)
                    buffer[index] = rgba
                sources = _rgba_source(buffer)
        done += count
        count = grid_count


def interpolate_frames(images: torch.Tensor, src_fps, multiplier: int, motion_mode: str = "auto",
                       scene_threshold: float = 0.24, runtime_dir: str = "",
                       wine_prefix: str = "", progress_callback=None,
                       target_fps: Fraction | None = None) -> tuple[torch.Tensor, Fraction, str]:
    """CFR samples from t=0 through the last source timestamp (no tail extension).

    N_out = floor((N_in - 1) * out_fps / src_fps) + 1. The last source is
    included only if the target timeline lands there; no off-grid tail append.
    CPU output and non-final cascade buffers scale with clip length.
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise DLSS5Error("DLSS frame interpolation expects (T, H, W, 3) frames.")
    count, height, width, _ = images.shape
    stages, out_fps, _ = _plan(src_fps, multiplier, target_fps)
    runtime, src_fps = _prepare(src_fps, width, height, count, runtime_dir)
    out_count = int((count - 1) * out_fps / src_fps) + 1
    result = torch.empty((out_count, height, width, 3), dtype=torch.float32)
    engines = []
    frames = _sampled_frames(_image_source(images), count, runtime, src_fps, stages,
                             out_fps, wine_prefix, scene_threshold, motion_mode,
                             progress_callback, engines)
    with closing(frames):
        for index, (rgba, original) in enumerate(frames):
            if original is not None:
                result[index] = original
            else:
                _rgba_into(result, index, rgba)
    note = (f"dlssg engine={engines[0]} {count}->{out_count} frames "
            f"@{out_fps}fps, {2 ** stages}x grid, no tail extension (runtime {runtime})")
    return result, out_fps, note


def interpolate_stream(rgb, src_fps, multiplier: int, motion_mode: str = "auto",
                       scene_threshold: float = 0.24, runtime_dir: str = "",
                       wine_prefix: str = "", progress_callback=None,
                       frame_sink=None, target_fps: Fraction | None = None) -> tuple[Fraction, str]:
    """Same timeline as IMAGE; encode selected final-stage frames immediately.

    Input RGB and non-final RGBA buffers still scale with clip length.
    """
    if frame_sink is None:
        raise DLSS5Error("interpolate_stream needs a frame_sink callable.")
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise DLSS5Error("DLSS frame interpolation expects (T, H, W, 3) frames.")
    count, height, width, _ = rgb.shape
    stages, out_fps, _ = _plan(src_fps, multiplier, target_fps)
    runtime, src_fps = _prepare(src_fps, width, height, count, runtime_dir)
    engines = []
    frames = _sampled_frames(_rgb_source(rgb), count, runtime, src_fps, stages,
                             out_fps, wine_prefix, scene_threshold, motion_mode,
                             progress_callback, engines)
    emitted = 0
    with closing(frames):
        for rgba, _ in frames:
            frame_sink(np.ascontiguousarray(rgba[..., :3]))
            emitted += 1
    note = (f"dlssg engine={engines[0]} {count}->{emitted} frames "
            f"@{out_fps}fps, {2 ** stages}x grid, no tail extension (runtime {runtime})")
    return out_fps, note
