# SPDX-License-Identifier: MIT
# Linux backend: stream frames to dlss5nr_host.exe under Wine.
# Transport protocol adapted from the MIT-licensed ComfyUI-DLSS5-NR-Linux
# frontend (kos94ok/ComfyUI-DLSS5-NR-Linux, tools/dlss5nr_video.py).

from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import common
from .common import DLSS5Error


def find_wine() -> str:
    """Wine binary resolution: DLSS5_WINE env -> WineHQ /opt/wine-stable -> PATH."""
    override = os.environ.get("DLSS5_WINE")
    if override and Path(override).is_file():
        return override
    winehq = "/opt/wine-stable/bin/wine"
    if Path(winehq).is_file():
        return winehq
    found = shutil.which("wine")
    if found:
        return found
    raise DLSS5Error(
        "Wine was not found on this machine. Install Wine (9.0+ recommended, e.g. "
        "WineHQ stable) or point DLSS5_WINE at a wine binary."
    )


_xvfb_lock = threading.Lock()
_xvfb_proc: subprocess.Popen | None = None
_xvfb_display: str | None = None


def _display_alive(candidate: int) -> bool:
    """True when an X server is accepting connections on the display's socket.

    Adopted-orphan case included: an Xvfb left behind by a SIGKilled worker is
    perfectly usable, so we reuse it instead of spawning another one.
    """
    lock = Path(f"/tmp/.X{candidate}-lock")
    sock = Path(f"/tmp/.X11-unix/X{candidate}")
    if not lock.exists() or not sock.exists():
        return False
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        probe.connect(str(sock))
        probe.close()
        return True
    except OSError:
        return False


def _reclaim_display(candidate: int) -> None:
    """Drop stale lock/socket files of a display whose server is gone."""
    for path in (Path(f"/tmp/.X{candidate}-lock"), Path(f"/tmp/.X11-unix/X{candidate}")):
        try:
            path.unlink()
        except OSError:
            pass


def _terminate_shared_xvfb() -> None:
    with _xvfb_lock:
        global _xvfb_proc
        proc, _xvfb_proc = _xvfb_proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def _start_xvfb_if_needed(env: dict):
    """Headless server: ensure some private Xvfb exists and point DISPLAY at it.

    One process-wide Xvfb is started on first use and reused by every later
    wine session (probes included); it is terminated at interpreter exit via
    atexit, so callers never own (and must never kill) the returned value.
    Displays left behind by SIGKilled workers are probed and adopted when the
    server still answers, or their stale lock/socket files are reclaimed.

    Running wine directly with DISPLAY set keeps the child process file
    descriptors clean; wrappers like xvfb-run have been observed to leak NGX
    log output onto the host's binary stdout stream.
    """
    global _xvfb_proc, _xvfb_display
    if env.get("DISPLAY") or not shutil.which("Xvfb"):
        return None
    with _xvfb_lock:
        if _xvfb_proc is not None and _xvfb_proc.poll() is None:
            env["DISPLAY"] = _xvfb_display
            return None
        for candidate in range(99, 160):
            if _display_alive(candidate):
                env["DISPLAY"] = f":{candidate}"
                return None  # adopted: not ours to terminate
            _reclaim_display(candidate)
            proc = subprocess.Popen(
                ["Xvfb", f":{candidate}", "-screen", "0", "1280x720x24"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                proc.wait(timeout=3)
                # Xvfb daemonises? It should stay in the foreground; if it exited
                # immediately the display number is taken or Xvfb failed.
                if proc.returncode not in (0, None):
                    continue
            except subprocess.TimeoutExpired:
                pass
            sock = Path(f"/tmp/.X11-unix/X{candidate}")
            deadline = 5.0
            while deadline > 0 and not sock.exists():
                time.sleep(0.1)
                deadline -= 0.1
            if sock.exists() and _display_alive(candidate):
                _xvfb_proc = proc
                _xvfb_display = f":{candidate}"
                env["DISPLAY"] = _xvfb_display
                return None
            try:
                proc.terminate()
            except Exception:
                pass
    return None


atexit.register(_terminate_shared_xvfb)


def _host_command(host: Path, runtime: Path, gpu_index: int, wine_prefix: str = "",
                  hdr: bool = False) -> tuple[list[str], dict]:
    wine = find_wine()
    env = os.environ.copy()
    env.setdefault("DLSS5NR_DISABLE_OTHER_SINKS", "1")
    env["DLSS5NR_GPU_INDEX"] = str(int(gpu_index))
    # The bridge reads this at carrier-feature creation: AutoExposure is always
    # on, IsHDR follows linear-light feeding (see dlss5nr_bridge.cpp).
    env["DLSS5NR_HDR"] = "1" if hdr else "0"
    env["DLSS5NR_SNR_FILENAME"] = common.resolve_snr_filename(runtime, gpu_index)
    env.setdefault("DXVK_ENABLE_NVAPI", "1")
    env.setdefault("DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE", "on")
    # Prefer DXVK-NVAPI / vkd3d-proton native DLLs, then fall back to Wine's
    # builtin D3D12. A builtin-only override hides the physical NVIDIA adapter
    # from NvAPI, which NGX requires.
    env.setdefault("WINEDLLOVERRIDES", "d3d12,d3d12core,nvapi64,dxgi=n,b")
    env.setdefault("WINEDEBUG", "-all")
    # The host stdout carries binary frames; DXVK/vkd3d log noise must never
    # reach it (some launch configurations send their logs to stdout).
    env.setdefault("DXVK_LOG_LEVEL", "none")
    env.setdefault("VKD3D_DEBUG", "none")
    if wine_prefix:
        env["WINEPREFIX"] = str(wine_prefix)

    cmd = [wine, str(host), str(runtime)]
    return cmd, env


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


def _read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise DLSS5Error("DLSS5 host closed the pipe unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class WineHostSession:
    """One dlss5nr_host.exe process owning a persistent D3D12/NGX session."""

    def __init__(self, runtime: Path, gpu_index: int = 0, wine_prefix: str = "",
                 hdr: bool = False) -> None:
        host = common.PLUGIN_ROOT / "native" / "bin" / "dlss5nr_host.exe"
        if not host.is_file():
            raise DLSS5Error(f"DLSS5 host executable is missing: {host}")
        self.cmd, self.env = _host_command(host, runtime, gpu_index, wine_prefix, hdr)
        self.cwd = str(common.PLUGIN_ROOT)
        self.runtime = runtime
        self.proc: subprocess.Popen | None = None
        self.drain: _StderrDrain | None = None
        _start_xvfb_if_needed(self.env)  # process-wide singleton; never killed per session

    def _start(self, input_w: int, input_h: int, output_w: int, output_h: int,
               warmup_frames: int, frame_count: int, perf_quality: int, params: dict) -> None:
        _start_xvfb_if_needed(self.env)
        try:
            self.proc = subprocess.Popen(
                self.cmd, cwd=self.cwd, env=self.env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise DLSS5Error(f"Could not start the DLSS5 Wine host ({self.cmd[0]}): {exc}") from exc
        self.drain = _StderrDrain(self.proc.stderr)
        self.drain.start()
        header = common.HEADER.pack(
            common.MAGIC_DNR2,
            int(input_w), int(input_h), int(output_w), int(output_h),
            int(warmup_frames), int(frame_count), int(perf_quality),
            0,  # profile
            int(params["preset"]),
            int(params["style"]),
            1 if params["auto_mask"] else 0,
            0,  # ui correction
            float(params["intensity"]),
            float(params["tone"]),
            float(params["structure"]),
            float(params["skin"]),
            float(params["global_tone"]),
        )
        try:
            self.proc.stdin.write(header)
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise DLSS5Error(f"DLSS5 host exited during handshake:\n{self.drain.tail(20)}") from exc

    def _fail(self, message: str) -> DLSS5Error:
        detail = self.drain.tail(20) if self.drain else ""
        return DLSS5Error(f"{message}\nDLSS5 host log:\n{detail}")

    def send_frame(self, index: int, source: np.ndarray, motion: np.ndarray, reset: bool) -> np.ndarray:
        """Send one RGB float32 frame, return the processed RGB float32 frame."""
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        input_h, input_w, _ = source.shape
        output_h, output_w, _ = self.output_shape
        try:
            self.proc.stdin.write(common.FRAME_HEADER.pack(common.MAGIC_FRM2, index, 1 if reset else 0))
            self.proc.stdin.write(np.ascontiguousarray(source, dtype=np.float32).tobytes(order="C"))
            self.proc.stdin.write(np.ascontiguousarray(motion, dtype=np.float16).view(np.uint16).tobytes(order="C"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._fail(f"DLSS5 pipeline I/O failed on frame {index}: {exc}") from exc

        try:
            magic, reply_index, ok, count = common.REPLY_HEADER.unpack(
                _read_exact(self.proc.stdout, common.REPLY_HEADER.size))
        except DLSS5Error as exc:
            raise self._fail(f"Invalid DLSS5 host reply for frame {index}: {exc}") from exc
        if magic != common.MAGIC_OUT1 or reply_index != index:
            raise self._fail(f"Invalid DLSS5 host reply for frame {index}: magic={magic!r}, index={reply_index}")
        if not ok:
            length = int(np.frombuffer(_read_exact(self.proc.stdout, 4), dtype="<u4")[0])
            detail = _read_exact(self.proc.stdout, min(length, 65535)).decode("utf-8", errors="replace")
            raise self._fail(f"DLSS5 frame {index} failed: {detail}")
        expected = output_w * output_h * 3
        if count != expected:
            raise self._fail(f"DLSS5 frame {index} returned {count} floats, expected {expected}")
        output = np.frombuffer(_read_exact(self.proc.stdout, expected * 4), dtype=np.float32)
        return output.reshape((output_h, output_w, 3))

    def finish(self) -> None:
        assert self.proc is not None
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
                self.proc.stdin = None
            done = _read_exact(self.proc.stdout, 4)
            if done != common.MAGIC_END1:
                raise self._fail("DLSS5 host did not send END1")
            # The 310.8 runtime can block in teardown after END1; frames are
            # already delivered, so do not let shutdown hang the queue.
            try:
                rc = self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.terminate()
                rc = 0
            if rc != 0:
                raise self._fail(f"DLSS5 host failed (exit {rc})")
        except DLSS5Error:
            raise
        finally:
            self.terminate()

    def terminate(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            # Kill the whole process group: wine spawns the host through
            # start.exe, so terminating only the wine loader would orphan the
            # actual host.exe process holding the GPU session.
            try:
                os.killpg(os.getpgid(proc.pid), 15)
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except Exception:
                    pass

    def __enter__(self) -> "WineHostSession":
        return self

    def __exit__(self, *exc) -> None:
        self.terminate()


_worker_lock = threading.Lock()


def process_frames(frames, input_w: int, input_h: int, output_w: int, output_h: int,
                   params: dict, progress_callback=None):
    """Run frames through the Wine host. `frames` yields (rgb_u8, motion, reset).

    Yields output RGB float32 frames (output_h, output_w, 3) in 0..1.
    """
    common.validate_sizes(input_w, input_h, output_w, output_h)
    session = WineHostSession(params["runtime"], params["gpu_index"], params.get("wine_prefix", ""),
                              bool(params.get("hdr")))
    session.output_shape = (output_h, output_w, 3)
    with _worker_lock:
        try:
            session._start(input_w, input_h, output_w, output_h,
                           params["warmup_frames"], params["frame_count"],
                           params["perf_quality"], params)
            for index, (pixels_u8, motion, reset) in enumerate(frames):
                source = pixels_u8.astype(np.float32) / 255.0
                if params.get("hdr"):
                    source = common.srgb_to_linear(source)
                out = session.send_frame(index, source, motion, reset)
                if params.get("detail", 1.0) != 1.0 or params.get("color", 1.0) != 1.0:
                    out = common.nr_composite(
                        pixels_u8, out, params["detail"], params["color"],
                        out_is_linear=bool(params.get("hdr")), encode=not params.get("hdr"),
                    )
                if params.get("hdr"):
                    out = common.linear_to_srgb(out)
                if progress_callback is not None:
                    progress_callback(index + 1, params["frame_count"])
                yield out
            session.finish()
        finally:
            session.terminate()
