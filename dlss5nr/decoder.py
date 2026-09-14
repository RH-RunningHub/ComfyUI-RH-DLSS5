# -*- coding: utf-8 -*-
"""Chunked video decoding for file-backed VIDEO inputs.

VIDEO.get_components() materialises the WHOLE clip as float32 (12 bytes per
pixel per frame): a 1824-frame 1080p clip costs ~45 GB RAM, a 4K clip ~181 GB
which the cgroup OOM-kills outright (20260914 incident on GZAS-175). When the
VIDEO is file-backed we can decode in bounded uint8 chunks through an ffmpeg
rawvideo pipe instead: peak memory is one chunk, regardless of clip length.

Readers are strictly sequential (frame N before N+1); both consumers in this
plugin (DLSS-FG stages, Enhance motion/derive loop) walk frames forward only,
and the SequentialSource asserts that so a protocol regression fails loudly
instead of silently duplicating frames.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


def _find_binary(name: str) -> str | None:
    return shutil.which(name)


def probe_video(path: str) -> dict | None:
    """ffprobe (read-only) -> {'frames','width','height','fps'} or None.

    frames prefers nb_read_packets (-count_packets: counts packets without
    decoding) and falls back to the container's nb_frames metadata.
    """
    ffprobe = _find_binary("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets,nb_frames,width,height,avg_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        streams = (json.loads(r.stdout or "{}").get("streams") or [])
    except Exception:
        return None
    if not streams:
        return None
    s = streams[0]

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    frames = _int(s.get("nb_read_packets")) or _int(s.get("nb_frames"))
    width, height = _int(s.get("width")), _int(s.get("height"))
    if frames <= 0 or width <= 0 or height <= 0:
        return None
    fps = 0.0
    ar = str(s.get("avg_frame_rate") or "")
    try:
        if "/" in ar and _int(ar.split("/", 1)[1]) != 0:
            fps = _int(ar.split("/", 1)[0]) / _int(ar.split("/", 1)[1])
        elif ar:
            fps = float(ar)
    except (TypeError, ValueError):
        fps = 0.0
    return {"frames": frames, "width": width, "height": height, "fps": fps}


class ChunkedVideoReader:
    """Stream a video file as sequential (index, uint8 RGB frame) tuples.

    One ffmpeg process reads the whole clip; memory peaks at one chunk
    (CHUNK_FRAMES frames of uint8 RGB) plus the decoder pipe, independent of
    clip length or resolution.
    """

    CHUNK_FRAMES = 32

    def __init__(self, path: str, log=print):
        self.path = str(path)
        self.log = log
        info = probe_video(self.path)
        if info is None:
            raise RuntimeError(f"cannot probe video for chunked decoding: {self.path}")
        self.count = int(info["frames"])
        self.width = int(info["width"])
        self.height = int(info["height"])
        self.fps = float(info["fps"])

    def iter_frames(self):
        """Yield (index, frame uint8 HxWx3 RGB) strictly in temporal order."""
        import numpy as np

        ffmpeg = _find_binary("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("no ffmpeg binary available for chunked decoding")
        w, h = self.width, self.height
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
               "-i", self.path, "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-v", "error", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_bytes = w * h * 3
        chunk = max(1, self.CHUNK_FRAMES)
        index = 0
        try:
            while index < self.count:
                raw = proc.stdout.read(frame_bytes * chunk)
                if not raw:
                    break
                got = len(raw) // frame_bytes
                if got <= 0:
                    break
                block = np.frombuffer(raw[:got * frame_bytes], dtype=np.uint8)
                frames = block.reshape(got, h, w, 3)
                for i in range(got):
                    if index >= self.count:
                        break
                    yield index, frames[i]
                    index += 1
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


class SequentialSource:
    """source(index) over a strictly forward frame iterator.

    Serves the same (rgba_u8, src_float) protocol as fg._rgb_source, but
    decodes lazily: memory is one frame ahead of the consumer. Non-monotonic
    or skipping access raises - the FG stage walk is 0..count-1 exactly once;
    any other pattern means someone changed the protocol and must handle
    random access explicitly.
    """

    def __init__(self, reader: ChunkedVideoReader):
        import numpy as np

        self._np = np
        self._reader = reader
        self._iter = reader.iter_frames()
        self._next_index = 0
        self._buffered = None   # (index, frame) decoded but not yet served
        self._served_up_to = -1

    def __call__(self, index: int):
        if index <= self._served_up_to:
            raise RuntimeError(
                f"chunked source non-monotonic access: index {index} already served "
                f"(served_up_to={self._served_up_to}); FG stage walks must be sequential")
        if index > self._next_index:
            raise RuntimeError(
                f"chunked source skipped access: wanted index {index}, next unread "
                f"is {self._next_index}; FG stage walks must be sequential")
        while self._next_index <= index:
            item = next(self._iter, None)
            if item is None:
                raise RuntimeError(
                    f"chunked source exhausted at index {index} "
                    f"(decoded {self._next_index}/{self._reader.count} frames)")
            served_index, frame = item
            self._buffered = frame
            self._next_index = served_index + 1
        self._served_up_to = index
        rgba = self._np.empty((self._reader.height, self._reader.width, 4), dtype=self._np.uint8)
        rgba[..., :3] = self._buffered
        rgba[..., 3] = 255
        self._buffered = None
        return rgba, None


def build_source(path: str, log=print):
    """File-backed VIDEO source -> (reader, SequentialSource) or None."""
    try:
        reader = ChunkedVideoReader(path, log=log)
        return reader, SequentialSource(reader)
    except Exception as e:
        log(f"[RH-DLSS5] chunked decode unavailable ({e}); falling back to full decode")
        return None
