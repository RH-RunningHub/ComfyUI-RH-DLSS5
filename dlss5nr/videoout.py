# SPDX-License-Identifier: MIT
# VIDEO output construction for ComfyUI-RH-DLSS5.
# Prefers the in-memory comfy_api.video path on current ComfyUI; the platform
# build instead exposes these classes under comfy_api.latest (InputImpl +
# Types.VideoComponents with frame_rate=Fraction(fps)). Falls back to a
# PyAV-encoded temporary file (audio muxed via ffmpeg when available) on
# older environments, and degrades to a status-only path when no VIDEO type
# exists at all.

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _video_from_components(images: torch.Tensor, fps: Fraction | float, audio_input):
    fps = Fraction(fps)
    try:
        from comfy_api.video import VideoFromComponents, VideoComponents  # type: ignore
    except Exception:
        pass
    else:
        config = None
        for module_name in ("comfy_api.input", "comfy_api.input_impl"):
            try:
                module = __import__(module_name, fromlist=["VideoAudioConfig"])
                config = module.VideoAudioConfig(fps=fps)
                if Fraction(config.fps) != fps:
                    config = None
                    continue
                break
            except Exception:
                continue
        try:
            if config is None:
                raise ValueError("No exact frame-rate config available")
            components = VideoComponents(images=images, audio=audio_input, config=config)
            return VideoFromComponents(components), None
        except Exception:
            pass
    # Platform build: VIDEO classes live under comfy_api.latest and
    # VideoComponents carries frame_rate instead of fps/config.
    try:
        from comfy_api.latest import InputImpl, Types  # type: ignore
        components = Types.VideoComponents(
            images=images, frame_rate=Fraction(fps), audio=audio_input
        )
        return InputImpl.VideoFromComponents(components), None
    except Exception:
        return None, None


def _encode_with_pyav(images: torch.Tensor, fps: Fraction | float, path: Path) -> None:
    import av

    _, height, width, _ = images.shape
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        stream.time_base = 1 / Fraction(fps)
        for index, frame_rgb in enumerate(images):
            frame = frame_rgb.detach().cpu().clamp(0.0, 1.0).numpy()
            frame_u8 = (frame * 255.0 + 0.5).astype(np.uint8)
            packet_source = av.VideoFrame.from_ndarray(frame_u8, format="rgb24")
            packet_source.pts = index
            packet_source.time_base = 1 / Fraction(fps)
            for packet in stream.encode(packet_source):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _audio_source_path(audio_input, source_video) -> str | None:
    for candidate in (source_video, audio_input):
        if candidate is None:
            continue
        getter = getattr(candidate, "get_stream_source", None)
        if callable(getter):
            try:
                info = getter()
                if isinstance(info, (tuple, list)) and info:
                    path = str(info[0])
                    if path and os.path.exists(path):
                        return path
                if isinstance(info, str) and os.path.exists(info):
                    return info
            except Exception:
                continue
    return None


def _mux_audio(video_only: Path, source: str, ffmpeg: str) -> Path:
    from folder_paths import get_temp_directory

    output = Path(get_temp_directory()) / f"dlss5_{uuid.uuid4().hex}_muxed.mp4"
    # A WAV fallback is PCM and cannot be stream-copied into mp4: re-encode it
    # (padded to the video length). Copied source tracks keep their duration;
    # -shortest is omitted so a shorter track can never truncate the video.
    if str(source).lower().endswith(".wav"):
        audio_opts = ["-c:a", "aac", "-af", "apad", "-shortest"]
    else:
        audio_opts = ["-c:a", "copy"]
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_only), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?",
        "-map_metadata", "1",
        "-c:v", "copy", *audio_opts,
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(result.stderr.strip()[-400:] or "ffmpeg audio mux failed")
    return output


def _video_from_file(path: Path):
    try:
        from comfy_api.video import VideoFromFile  # type: ignore

        return VideoFromFile(str(path))
    except Exception:
        pass
    try:
        from comfy_api.latest import InputImpl  # type: ignore

        return InputImpl.VideoFromFile(str(path))
    except Exception:
        return None


def video_from_file(path: Path):
    return _video_from_file(path)


def audio_source_path(audio_input, source_video) -> str | None:
    return _audio_source_path(audio_input, source_video)


def _audio_to_wav(audio_input) -> Path | None:
    """AUDIO object (waveform tensor + sample_rate) -> int16 WAV temp file.

    Fallback for when the audio cannot be stream-copied from a source file
    (e.g. an in-memory VIDEO whose stream source is unavailable). Returns None
    when the input carries no usable waveform.
    """
    wave = getattr(audio_input, "waveform", None)
    rate = getattr(audio_input, "sample_rate", None)
    if wave is None and isinstance(audio_input, dict):
        wave = audio_input.get("waveform")
        rate = audio_input.get("sample_rate")
    if wave is None or rate is None:
        return None
    try:
        import wave as wavefile

        from folder_paths import get_temp_directory

        samples = torch.as_tensor(wave).detach().cpu().float()
        if samples.ndim == 3:  # (batch, channels, samples)
            samples = samples[0]
        if samples.ndim == 1:
            samples = samples[None, :]
        pcm = (np.clip(samples.T.contiguous().numpy(), -1.0, 1.0) * 32767.0).astype("<i2")
        path = Path(get_temp_directory()) / f"dlss5_{uuid.uuid4().hex}_audio.wav"
        with wavefile.open(str(path), "wb") as handle:
            handle.setnchannels(pcm.shape[1])
            handle.setsampwidth(2)
            handle.setframerate(int(rate))
            handle.writeframes(pcm.tobytes())
        return path
    except Exception:
        return None


class StreamEncoder:
    """Incremental RGB-uint8 sink that pipes frames into an ffmpeg rawvideo encoder.

    Peak memory is one frame plus the encoder process; the mp4 lands in the
    ComfyUI temp directory and is wrapped as a VIDEO file object afterwards.
    Audio (when the source file is locatable) is stream-copied by the same
    ffmpeg invocation. Lifecycle: write() per frame, then close(); on any
    exception call abort() instead.
    """

    def __init__(self, width: int, height: int, fps, audio_input=None,
                 source_video=None) -> None:
        fps = Fraction(fps)
        self.width = int(width)
        self.height = int(height)
        self.source = None
        self._audio_wav: Path | None = None
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("no ffmpeg binary available for streaming video encode")
        from folder_paths import get_temp_directory

        self.path = Path(get_temp_directory()) / f"dlss5_{uuid.uuid4().hex}.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", f"{fps.numerator}/{fps.denominator}",
            "-i", "-",
        ]
        if audio_input is not None:
            source = _audio_source_path(audio_input, source_video)
            if source:
                self.source = source
                command += ["-i", source, "-map", "0:v:0", "-map", "1:a?",
                            "-c:a", "copy", "-shortest"]
            else:
                # No source file to copy from: re-encode the AUDIO object itself.
                # apad + -shortest pads short audio with silence so the VIDEO
                # length always wins (a shorter track must not truncate it).
                self._audio_wav = _audio_to_wav(audio_input)
                if self._audio_wav is not None:
                    command += ["-i", str(self._audio_wav), "-map", "0:v:0", "-map", "1:a",
                                "-c:a", "aac", "-af", "apad", "-shortest"]
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(self.path)]
        self._proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._errors: list[str] = []
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain.start()

    def _drain_stderr(self) -> None:
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                self._errors.append(raw.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

    def _error_tail(self) -> str:
        return "\n".join(self._errors[-8:])

    def write(self, rgb) -> None:
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[0] != self.height or frame.shape[1] != self.width \
                or frame.shape[2] != 3:
            raise RuntimeError(f"stream encoder got frame shape {tuple(frame.shape)}, "
                               f"expected ({self.height}, {self.width}, 3)")
        try:
            self._proc.stdin.write(frame.tobytes())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"ffmpeg encoder died mid-stream:\n{self._error_tail()}\n{exc}")

    def close(self) -> None:
        proc = self._proc
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            rc = proc.wait(timeout=300)
        finally:
            try:
                proc.stderr.close()
            except OSError:
                pass
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder exited {rc}:\n{self._error_tail()}")
        self._cleanup_audio_wav()

    def abort(self) -> None:
        proc = self._proc
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        self._cleanup_audio_wav()

    def _cleanup_audio_wav(self) -> None:
        wav, self._audio_wav = getattr(self, "_audio_wav", None), None
        if wav is not None:
            try:
                wav.unlink(missing_ok=True)
            except OSError:
                pass


def build_video(images: torch.Tensor, fps: Fraction | float, audio_input=None, source_video=None) -> tuple[object | None, str]:
    """Build a ComfyUI VIDEO object from processed frames.

    Returns (video_or_None, status_note). audio_input is the original
    components.audio object when available; source_video is the upstream VIDEO
    input used to locate the original file for audio stream copy.
    """
    # `_run_frames` already delivers [0,1] contiguous CPU frames; only copy or
    # clamp when actually needed so long videos don't get buffered twice.
    images = images.detach().cpu()
    if not images.is_contiguous():
        images = images.contiguous()
    if images.numel() and (float(images.min()) < 0.0 or float(images.max()) > 1.0):
        images.clamp_(0.0, 1.0)

    video_obj, _ = _video_from_components(images, fps, audio_input)
    if video_obj is not None:
        return video_obj, "in-memory VIDEO object (audio preserved when provided)"

    from folder_paths import get_temp_directory

    temp_path = Path(get_temp_directory()) / f"dlss5_{uuid.uuid4().hex}.mp4"
    _encode_with_pyav(images, fps, temp_path)

    audio_note = "no audio"
    if audio_input is not None:
        ffmpeg = _find_ffmpeg()
        source = _audio_source_path(audio_input, source_video)
        wav = None
        if ffmpeg and not source:
            # No copyable source track: re-encode the AUDIO waveform instead
            # of dropping sound entirely (matches the streaming encoder).
            wav = _audio_to_wav(audio_input)
            source = str(wav) if wav else None
        if ffmpeg and source:
            try:
                muxed = _mux_audio(temp_path, source, ffmpeg)
                if muxed != temp_path:
                    temp_path.unlink(missing_ok=True)
                    temp_path = muxed
                audio_note = ("audio stream-copied from source" if wav is None
                              else "audio re-encoded from AUDIO input")
            except Exception as exc:  # keep the video, report the loss
                audio_note = f"audio mux failed ({exc}); output is video-only"
        else:
            audio_note = "audio dropped (no ffmpeg binary or source file available)"
        if wav is not None:
            try:
                wav.unlink(missing_ok=True)
            except OSError:
                pass

    video_obj = _video_from_file(temp_path)
    note = f"temp file {temp_path} ({audio_note})"
    if video_obj is None:
        note = f"this ComfyUI build has no VIDEO type; processed file written to {temp_path} ({audio_note})"
    return video_obj, note
