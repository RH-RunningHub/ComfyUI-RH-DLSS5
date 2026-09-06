# SPDX-License-Identifier: MIT
# ComfyUI-RH-DLSS5: single-node DLSS 5 Neural Rendering (NGX feature 18)
# enhance / upscale for IMAGE and VIDEO, Windows (in-process bridge) and
# Linux (Wine host) backends.

from __future__ import annotations

import hashlib
import os
import platform
from typing import Optional

import numpy as np
import torch

from .dlss5nr import common
from .dlss5nr.common import DLSS5Error
from .dlss5nr.motion import TemporalGuideGenerator, sample_u8

_UPSCALE_MODES = [
    "1x (DLAA / native)",
    "1.5x (Quality)",
    "1.724x (Balanced)",
    "2x (Performance)",
    "3x (Ultra Performance)",
]

_SCALE_KEY = {"1x": 1.0, "1.5x": 1.5, "1.724x": 1.724, "2x": 2.0, "3x": 3.0}


def _load_backend(name: str):
    if name == "windows-bridge":
        from .dlss5nr import windows_backend

        if not windows_backend.supported():
            raise DLSS5Error("The windows-bridge backend requires Windows 10/11 x64.")
        return windows_backend
    from .dlss5nr import linux_backend

    if platform.system() == "Windows":
        raise DLSS5Error(
            "The linux-wine backend requires Linux; on Windows use backend=windows-bridge."
        )
    return linux_backend


def _pick_backend(backend: str) -> str:
    if backend == "auto":
        return "windows-bridge" if platform.system() == "Windows" else "linux-wine"
    return backend


def _channel_choice(frame: np.ndarray, reference: np.ndarray, order: str) -> tuple[np.ndarray, str]:
    """Detect R/B swap of the NR output (runtime builds differ). Adapted from
    the MIT-licensed ComfyUI-DLSS5-NR-Linux frontend."""
    if order == "RGBA":
        return frame, "RGBA"
    if order == "BGRA":
        return frame[..., ::-1], "BGRA"
    sample_h = min(128, frame.shape[0], reference.shape[0])
    sample_w = min(128, frame.shape[1], reference.shape[1])
    iy = np.linspace(0, frame.shape[0] - 1, sample_h, dtype=np.int64)
    ix = np.linspace(0, frame.shape[1] - 1, sample_w, dtype=np.int64)
    ry = np.linspace(0, reference.shape[0] - 1, sample_h, dtype=np.int64)
    rx = np.linspace(0, reference.shape[1] - 1, sample_w, dtype=np.int64)
    raw = frame[np.ix_(iy, ix)]
    ref = reference[np.ix_(ry, rx)]
    swapped = raw[..., ::-1]
    raw_score = float(np.mean(np.abs(raw - ref))) + float(
        np.mean(np.abs(raw.mean(axis=(0, 1)) - ref.mean(axis=(0, 1)))))
    swap_score = float(np.mean(np.abs(swapped - ref))) + float(
        np.mean(np.abs(swapped.mean(axis=(0, 1)) - ref.mean(axis=(0, 1)))))
    if raw_score <= swap_score:
        return frame, "RGBA"
    return frame[..., ::-1], "BGRA"


def _build_params(kwargs: dict) -> dict:
    mode = kwargs.get("upscaling_mode") or _UPSCALE_MODES[0]
    scale = _SCALE_KEY[next(key for key in _SCALE_KEY if str(mode).startswith(key))]
    runtime = common.resolve_runtime_dir(kwargs.get("runtime_dir", "") or "")
    common.check_runtime_files(runtime, scale)
    # GPU selection is an environment concern (queue schedulers pin one worker
    # per device); it is deliberately not a widget.
    try:
        gpu_index = int(os.environ.get("DLSS5_GPU_INDEX", "0") or 0)
    except ValueError:
        gpu_index = 0
    return {
        "runtime": runtime,
        "gpu_index": gpu_index,
        "wine_prefix": str(kwargs.get("wine_prefix", "") or ""),
        "scale": scale,
        "preset": int(kwargs.get("preset", 0)),
        "style": common.style_int(kwargs.get("style", "default")),
        "intensity": float(kwargs.get("intensity", 1.0)),
        "tone": float(kwargs.get("tone", 1.0)),
        "structure": float(kwargs.get("structure", 1.5)),
        "skin": float(kwargs.get("skin", -1.0)),
        "auto_mask": kwargs.get("auto_mask", "on") == "on",
        "warmup_frames": int(kwargs.get("warmup_frames", 0)),
        "keep_audio": kwargs.get("keep_audio", "on") == "on",
        "motion_mode": kwargs.get("motion", "auto"),
        "scene_change_threshold": float(kwargs.get("scene_change_threshold", 0.24)),
        "batch_temporal": kwargs.get("batch_mode", "temporal sequence") == "temporal sequence",
        "channel_order": kwargs.get("channel_order", "auto"),
        "hdr": kwargs.get("hdr", "off") == "on",
        "global_tone": float(kwargs.get("global_tone", -1.0)),
        "detail": float(kwargs.get("detail", 1.0)),
        "color": float(kwargs.get("color", 1.0)),
    }


def _make_nvof_generator(width: int, height: int, threshold: float):
    try:
        from .dlss5nr.nvof import NvofGuideGenerator
    except Exception:
        return None
    return NvofGuideGenerator(width, height, threshold)


def _select_motion_engine(pixels_u8: np.ndarray, params: dict):
    """Pick the guide generator for this run.

    Returns (generator, engine_note). The generator is None when only zero
    motion is possible; the note names the engine actually used, including
    any fallback that happened, and goes into the node's status output.
    """
    mode = params["motion_mode"]
    if TemporalGuideGenerator is None:
        return None, "none"
    height, width = pixels_u8.shape[:2]
    threshold = params["scene_change_threshold"]
    nvof_note = ""
    if mode in ("auto", "nvof"):
        generator = _make_nvof_generator(width, height, threshold)
        if generator is not None and generator.available:
            return generator, "nvof"
        nvof_note = (
            f"nvof unavailable ({generator.init_error})"
            if generator is not None
            else "nvof unavailable (module import failed)"
        )
        print(f"[RH-DLSS5] {nvof_note}; falling back to DIS optical flow", flush=True)
    generator = TemporalGuideGenerator(width, height, threshold)
    if generator.available:
        return generator, "nvof->optical_flow" if nvof_note else "optical_flow"
    if nvof_note:
        print("[RH-DLSS5] DIS optical flow unavailable too; zero motion", flush=True)
    return None, "none"


def _frame_source(images: torch.Tensor, params: dict):
    """Yield (rgb_u8, motion_fp16, reset) for every input frame."""
    temporal = params["batch_temporal"]
    flow_requested = params["motion_mode"] in ("auto", "nvof", "optical_flow")
    guides = None
    zero_motion: Optional[np.ndarray] = None
    seen_first = False
    for index in range(images.shape[0]):
        pixels_u8 = sample_u8(images[index].detach().cpu().numpy())
        if flow_requested and guides is None:
            guides, engine_note = _select_motion_engine(pixels_u8, params)
            params["motion_engine"] = engine_note
        if guides is not None and guides.available:
            motion, guide_reset = guides.process(pixels_u8)
            reset = guide_reset or not temporal
        else:
            if zero_motion is None:
                zero_motion = np.zeros((pixels_u8.shape[0], pixels_u8.shape[1], 2), dtype=np.float16)
            motion = zero_motion
            reset = (not temporal) or (not seen_first)
            seen_first = True
        yield pixels_u8, motion, reset


def _progress_callback():
    try:
        from comfy.utils import ProgressBar

        bar = ProgressBar(100)

        def callback(current: int, frames_total: int) -> None:
            bar.update_absolute(int(100 * current / max(1, frames_total)))

        return callback
    except Exception:
        return None


class RH_DLSS5Enhance:
    """DLSS 5 Neural Rendering enhance/upscale node.

    IMAGE in -> IMAGE out, VIDEO in -> VIDEO out. Connect the outputs to the
    standard SaveImage / SaveVideo nodes. Only one input needs to be connected;
    if both are connected both are processed.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 (ComfyUI convention)
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image batch (e.g. LoadImage). Each batch entry is one frame in temporal order."}),
                "video": ("VIDEO", {"tooltip": "Optional video (e.g. LoadVideo). Frames are processed in temporal order."}),
                "upscaling_mode": (_UPSCALE_MODES, {"default": _UPSCALE_MODES[0], "tooltip": "1x enhances at native size; other modes upscale with the DLSS carrier (feature 1) before the neural pass (feature 18). Fixed NVIDIA factors only."}),
                "style": (["default", "natural", "cinematic", "off (bypass NR)"], {"default": "default", "tooltip": "Neural rendering style. Natural stays closer to the source; cinematic pushes contrast. off (bypass NR) disables all processing: frames pass through untouched, upscaling included - use it when you want the original look or zero processing cost."}),
                "preset": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1, "tooltip": "Internal render preset hint (DLSSNR.Hint.Render.Preset). Keep 0 unless the runtime documents another value."}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Neural pass strength. Above 1.0 usually has no extra effect; below 1.0 blends towards the source."}),
                "tone": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Local tone mapping strength (DLSSNR.LocalToneStrength)."}),
                "structure": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Local detail / structure reconstruction strength (DLSSNR.LocalStructureStrength)."}),
                "skin": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, "tooltip": "Skin structure strength (DLSSNR.SkinStructureStrength). Only active with auto_mask on. Negative leaves the parameter at the model default (the reference tool's -1)."}),
                "auto_mask": (["on", "off"], {"default": "on", "tooltip": "Let the model detect skin regions it reconstructs. Gates the skin parameter."}),
                "motion": (["auto", "nvof", "optical_flow", "none"], {"default": "auto", "tooltip": "Motion vectors for temporal accumulation. auto = hardware NVOFA optical flow when available, else DIS optical flow, else zero motion; nvof = hardware NVOFA (falls back to DIS with a note); optical_flow = OpenCV DIS; none = zero motion."}),
                "scene_change_threshold": ("FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Mean luminance change above which temporal history resets (scene cut detection). Optical flow only."}),
                "batch_mode": (["temporal sequence", "still images"], {"default": "temporal sequence", "tooltip": "still images resets temporal history for every frame; temporal sequence keeps history between frames (video / coherent batches)."}),
                "warmup_frames": ("INT", {"default": 0, "min": 0, "max": 120, "step": 1, "tooltip": "Warm-up budget reported to the worker. 0 is fine; the first frame initialises temporal history via reset."}),
                "keep_audio": (["on", "off"], {"default": "on", "tooltip": "Keep the original audio in the VIDEO output. Audio is stream-copied from the source file; when no source file is available it is re-encoded from the AUDIO input instead."}),
                "backend": (["auto", "linux-wine", "windows-bridge"], {"default": "auto", "tooltip": "auto picks the native backend for the OS. linux-wine streams frames to dlss5nr_host.exe under Wine; windows-bridge runs the D3D12 bridge in-process."}),
                "channel_order": (["auto", "RGBA", "BGRA"], {"default": "auto", "tooltip": "Channel layout of the NR output. auto detects R/B swap on the first frame; force RGBA/BGRA if colours look wrong."}),
                "hdr": (["off", "on"], {"default": "off", "tooltip": "on feeds linear light to the neural model instead of the sRGB proxy (matches the reference --nr-hdr; higher-fidelity tonal response, costs a colour-space roundtrip). off keeps the default sRGB-encoded path. Existing workflows default to off."}),
                "global_tone": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, "tooltip": "Global tone strength (DLSSNR.GlobalToneStrength) - the knob that lets the model shift overall brightness (the reference tool's global_tone). -1 leaves it at the model default; 0 disables global tone; 0-2 overrides manually. No-op on the 310.8 test runtime (verified); may act on other DLL builds."}),
                "detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Composite strength over the original frame (reference tool's detail): 0 = keep the original, 1 = full NR result, up to 2 = amplify the model's change (brightening/denoise) beyond 100%. Blended in linear light."}),
                "color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Chroma follow (reference tool's color): 0 = keep the original hue and take only the model's luminance, 1 = take the model's colour fully. Guards against the model shifting colours."}),
                "runtime_dir": ("STRING", {"default": "", "tooltip": "Optional override of the runtime folder holding nvngx_dlssnr.dll (and nvngx_dlss.dll for >1x). Empty uses DLSS5_RUNTIME_DIR, then the plugin's runtime/ folder."}),
                "wine_prefix": ("STRING", {"default": "", "tooltip": "Linux only: WINEPREFIX for the worker. Empty keeps the environment default. The prefix needs DXVK-NVAPI + vkd3d-proton for NVIDIA NGX."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING")
    RETURN_NAMES = ("images", "video", "status")
    FUNCTION = "enhance"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "Runs NVIDIA DLSS 5 Neural Rendering (NGX feature 18) over image batches or video frames, "
        "with optional 1.5x-3x neural upscaling via the ordinary DLSS carrier. Windows uses an "
        "in-process D3D12 bridge; Linux streams frames to dlss5nr_host.exe under Wine. Connect the "
        "outputs to the standard SaveImage / SaveVideo nodes. NVIDIA runtime DLLs are user-supplied "
        "and never redistributed."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        digest = hashlib.sha256()
        for key in sorted(kwargs):
            if key in ("image", "video"):
                continue
            digest.update(f"{key}={kwargs.get(key)!r};".encode())
        image = kwargs.get("image")
        if image is not None:
            tensor = image if isinstance(image, torch.Tensor) else image[0]
            digest.update(str(tuple(tensor.shape)).encode())
            flat = tensor.detach().reshape(-1)[:: max(1, tensor.numel() // 4096)]
            digest.update(flat.to(torch.float16).cpu().numpy().tobytes())
        video = kwargs.get("video")
        if video is not None:
            source = _audio_source_path_of(video)
            if source:
                try:
                    stat = os.stat(source)
                    digest.update(f"{source}:{stat.st_size}:{stat.st_mtime}".encode())
                except OSError:
                    digest.update(source.encode())
            else:
                digest.update(repr(video).encode())
        return digest.hexdigest()

    def _bypass_passthrough(self, image, video, kwargs) -> tuple[Optional[torch.Tensor], object, str]:
        """style=off: neural rendering disabled, frames pass through untouched."""
        from .dlss5nr.videoout import build_video

        note = "NR bypassed (style=off): frames passed through unprocessed"
        out_image: Optional[torch.Tensor] = None
        out_video = None
        if image is not None:
            out_image = image
            video_obj, _ = build_video(image, 24.0, None, None)
            if video_obj is not None:
                out_video = video_obj
            note = f"IMAGE {int(image.shape[0])} frames; {note}"
        if video is not None:
            components = video.get_components()
            frames = components.images
            config = getattr(components, "config", None)
            fps = float(getattr(config, "fps", 24.0) or 24.0) if config is not None else 24.0
            video_obj, _ = build_video(
                frames, fps, components.audio if str(kwargs.get("keep_audio", "on")) == "on" else None,
                video,
            )
            out_video = video_obj
            note = f"VIDEO {int(frames.shape[0])} frames @{fps:g}fps; {note}"
        return out_image, out_video, note

    def enhance(self, **kwargs) -> tuple[Optional[torch.Tensor], object, str]:
        image = kwargs.get("image")
        video = kwargs.get("video")
        if image is None and video is None:
            raise DLSS5Error("RH DLSS5 Enhance needs at least one input: connect `image` or `video`.")
        if str(kwargs.get("style", "default")).startswith("off"):
            return self._bypass_passthrough(image, video, kwargs)
        backend_name = _pick_backend(kwargs.get("backend", "auto"))
        params = _build_params(kwargs)
        gt_note = f", gt={params['global_tone']:g}" if params["global_tone"] >= 0 else ""

        out_image: Optional[torch.Tensor] = None
        out_video = None
        notes: list[str] = []

        if image is not None:
            result, iw, ih, ow, oh, order, engine = _run_frames(image, params, backend_name)
            out_image = result
            notes.append(
                f"IMAGE {int(image.shape[0])} frames {iw}x{ih}->{ow}x{oh} via {backend_name} "
                f"(channel {order}, motion {engine}{', hdr' if params['hdr'] else ''}{gt_note})"
            )

        if video is not None:
            components = video.get_components()
            frames = components.images
            config = getattr(components, "config", None)
            fps = float(getattr(config, "fps", 24.0) or 24.0) if config is not None else 24.0
            keep_audio = params["keep_audio"]
            result, iw, ih, ow, oh, order, engine = _run_frames(frames, params, backend_name)
            from .dlss5nr.videoout import build_video

            video_obj, note = build_video(
                result, fps, components.audio if keep_audio else None, video
            )
            out_video = video_obj
            notes.append(
                f"VIDEO {int(frames.shape[0])} frames @{fps:g}fps {iw}x{ih}->{ow}x{oh} via {backend_name} "
                f"(channel {order}, motion {engine}{', hdr' if params['hdr'] else ''}{gt_note}); {note}"
            )

        return out_image, out_video, " | ".join(notes)


class RH_DLSS5FrameInterpolation:
    """NVIDIA DLSS frame generation (NGX DLSS-FG) over video or image batches.

    Streams frames to the dlssg-worker.exe NGX host under Wine and interleaves
    the generated frames between the sources (2x = one generated frame per
    interval; 3x/4x cascade additional 2x passes). Frame count out is
    multiplier*T - (multiplier - 1): the final interval has no future frame to
    interpolate towards. The worker runtime is user-supplied, like the NVIDIA
    DLLs - see runtime/README.txt.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 (ComfyUI convention)
        return {
            "required": {},
            "optional": {
                "output_fps": (["2x", "3x", "4x", "23.976", "24", "25", "29.97", "30", "48", "50", "59.94", "60", "72", "90", "96", "120", "144"], {"default": "2x", "tooltip": "Either a frame-rate multiplier (2x/3x/4x: DLSS generates one frame per interval, 3x/4x cascade 2x passes) or an exact target output frame rate. For a target rate the node cascades DLSS 2x passes into a dense CFR grid (2/4/8x source) and nearest-picks the target timeline, so e.g. a 24fps source at 60fps interpolates a 96fps grid and picks every ~1.6th frame (1-2-1-2 cadence, no duplicated frames). Target rates must exceed source fps and be at most 6x source; intermediate grid is at most 8x. No tail extension."}),
                "video": ("VIDEO", {"tooltip": "Optional video (e.g. LoadVideo). Source frame rate and audio come from this input."}),
                "image": ("IMAGE", {"tooltip": "Optional image batch (e.g. LoadImage). Each batch entry is one frame in temporal order; batches are treated as a 24fps source."}),
                "motion": (["auto", "nvof", "dis"], {"default": "auto", "tooltip": "Motion vectors guiding interpolation. auto = hardware NVOFA optical flow when available, else OpenCV DIS; nvof = hardware NVOFA only; dis = OpenCV DIS. Scene cuts or disabled generation hold the previous frame in each interpolation slot."}),
                "scene_change_threshold": ("FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Mean luminance change above which temporal history resets (scene cut detection). Higher = fewer resets."}),
                "keep_audio": (["on", "off"], {"default": "on", "tooltip": "Keep the original audio in the VIDEO output. Audio is stream-copied from the source file; when no source file is available it is re-encoded from the AUDIO input instead."}),
                "audio": ("AUDIO", {"tooltip": "Optional audio for the VIDEO output. With a VIDEO input it replaces the source audio while keep_audio is on; with an IMAGE input it is the only way to attach sound. Re-encoded to AAC when it cannot be stream-copied."}),
                "runtime_dir": ("STRING", {"default": "", "tooltip": "Optional override of the folder holding dlssg-worker.exe + nvngx.dll + _nvngx.dll + nvngx_dlssg.dll. Empty uses DLSS5_FG_RUNTIME_DIR, then <DLSS5 runtime>/dlssg, then the plugin's runtime/dlssg folder."}),
                "wine_prefix": ("STRING", {"default": "", "tooltip": "Linux only: WINEPREFIX for the worker. Empty keeps the environment default. The prefix needs DXVK + vkd3d-proton + DXVK-NVAPI for NVIDIA NGX."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING")
    RETURN_NAMES = ("images", "video", "status")
    FUNCTION = "interpolate"
    CATEGORY = "video"
    DESCRIPTION = (
        "Raises a video's frame rate with NVIDIA DLSS frame generation (NGX DLSS-FG): "
        "generated frames are interleaved between the source frames instead of blended. "
        "Linux streams frames to dlssg-worker.exe under Wine; the worker runtime is "
        "user-supplied and never redistributed. Connect the outputs to the standard "
        "SaveImage / SaveVideo nodes."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        digest = hashlib.sha256()
        for key in sorted(kwargs):
            if key in ("image", "video", "audio"):
                continue
            digest.update(f"{key}={kwargs.get(key)!r};".encode())
        image = kwargs.get("image")
        if image is not None:
            tensor = image if isinstance(image, torch.Tensor) else image[0]
            digest.update(str(tuple(tensor.shape)).encode())
            flat = tensor.detach().reshape(-1)[:: max(1, tensor.numel() // 4096)]
            digest.update(flat.to(torch.float16).cpu().numpy().tobytes())
        audio = kwargs.get("audio")
        if audio is not None:
            wave = audio.get("waveform") if isinstance(audio, dict) else getattr(audio, "waveform", None)
            rate = audio.get("sample_rate") if isinstance(audio, dict) else getattr(audio, "sample_rate", None)
            if wave is not None:
                tensor = torch.as_tensor(wave)
                digest.update(f"audio:{rate}:{tuple(tensor.shape)}".encode())
                flat = tensor.detach().reshape(-1)[:: max(1, tensor.numel() // 4096)]
                digest.update(flat.to(torch.float16).cpu().numpy().tobytes())
        video = kwargs.get("video")
        if video is not None:
            source = _audio_source_path_of(video)
            if source:
                try:
                    stat = os.stat(source)
                    digest.update(f"{source}:{stat.st_size}:{stat.st_mtime}".encode())
                except OSError:
                    digest.update(source.encode())
            else:
                digest.update(repr(video).encode())
        return digest.hexdigest()

    def interpolate(self, **kwargs) -> tuple[Optional[torch.Tensor], object, str]:
        import gc

        from .dlss5nr.fg import MULTIPLIERS, as_fps, interpolate_frames, interpolate_stream
        from .dlss5nr.videoout import StreamEncoder, build_video, video_from_file

        image = kwargs.get("image")
        video = kwargs.get("video")
        if image is None and video is None:
            raise DLSS5Error("RH DLSS5 Frame Interpolation needs at least one input: connect `image` or `video`.")
        choice = str(kwargs.get("output_fps", "2x") or "2x")
        if choice in MULTIPLIERS:
            multiplier, target_fps = MULTIPLIERS[choice], None
        else:
            multiplier, target_fps = 2, as_fps(choice)
        motion_mode = str(kwargs.get("motion", "auto"))
        threshold = float(kwargs.get("scene_change_threshold", 0.24) or 0.24)
        keep_audio = str(kwargs.get("keep_audio", "on")) == "on"
        audio_input = kwargs.get("audio")
        runtime_dir = str(kwargs.get("runtime_dir", "") or "")
        wine_prefix = str(kwargs.get("wine_prefix", "") or "")

        from comfy.utils import ProgressBar

        out_image: Optional[torch.Tensor] = None
        out_video = None
        notes: list[str] = []

        if image is not None:
            src_fps = as_fps(kwargs.get("images_fps", 24.0))
            bar = ProgressBar(100)
            result, out_fps, note = interpolate_frames(
                image, src_fps, multiplier, motion_mode, threshold, runtime_dir, wine_prefix,
                progress_callback=lambda done, total: bar.update_absolute(100 * done // max(1, total)),
                target_fps=target_fps,
            )
            out_image = result
            notes.append(f"IMAGE {int(image.shape[0])}->{int(result.shape[0])} frames "
                         f"@{float(src_fps):g}->{float(out_fps):g}fps; {note}")
            video_obj, vnote = build_video(result, out_fps, audio_input, None)
            if video_obj is not None:
                out_video = video_obj
            if vnote:
                notes.append(vnote)

        if video is not None:
            components = video.get_components()
            frames = components.images
            config = getattr(components, "config", None)
            if config is not None and getattr(config, "fps", None):
                src_fps = as_fps(config.fps)
            else:
                frame_rate = getattr(components, "frame_rate", None)
                src_fps = as_fps(frame_rate if frame_rate is not None else 24)
            if target_fps is not None and target_fps <= src_fps:
                # Reject before decoding the whole clip: interpolate_stream
                # would only validate after the uint8 decode and encoder spawn.
                raise DLSS5Error(
                    f"target output fps ({float(target_fps):g}) must exceed the source "
                    f"rate ({float(src_fps):g}); interpolation only increases frame rate.")
            total_in = int(frames.shape[0])
            height, width = int(frames.shape[1]), int(frames.shape[2])
            if not keep_audio:
                audio = None
            elif audio_input is not None:
                audio = audio_input  # explicit AUDIO input replaces the source track
            else:
                audio = components.audio

            # Convert the decoded float batch to uint8 once, in bounded chunks,
            # then release local float references: RGB uint8 is one quarter
            # of float32 RGB. Upstream owners and cascade buffers may remain.
            rgb = np.empty((total_in, height, width, 3), dtype=np.uint8)
            chunk = max(1, (1 << 26) // max(1, height * width))
            for start in range(0, total_in, chunk):
                stop = min(total_in, start + chunk)
                block = frames[start:stop].detach().cpu().mul(255.0).round_().clamp_(0.0, 255.0)
                rgb[start:stop] = block.to(torch.uint8).numpy()
            frames = None
            try:
                components.images = None
            except Exception:
                pass
            gc.collect()

            # Stream the interpolation straight into the encoder: no full
            # float32 output tensor is ever built for the video path.
            out_fps = target_fps if target_fps is not None else src_fps * multiplier
            # With an explicit AUDIO input the source file must not win the
            # stream-copy race in _audio_source_path; drop it so the user
            # waveform is what gets encoded.
            encoder = StreamEncoder(width, height, out_fps, audio,
                                    video if audio_input is None else None)
            bar = ProgressBar(100)
            try:
                out_fps, note = interpolate_stream(
                    rgb, src_fps, multiplier, motion_mode, threshold, runtime_dir, wine_prefix,
                    progress_callback=lambda done, total: bar.update_absolute(100 * done // max(1, total)),
                    frame_sink=encoder.write, target_fps=target_fps,
                )
                encoder.close()
            except BaseException:
                encoder.abort()
                raise
            out_video = video_from_file(encoder.path)
            if encoder.source:
                audio_note = "audio stream-copied from source"
            elif audio_input is not None:
                audio_note = "audio re-encoded from audio input"
            else:
                audio_note = "no audio stream"
            notes.append(f"VIDEO {total_in} source frames "
                         f"@{float(src_fps):g}->{float(out_fps):g}fps, streaming encode, "
                         f"{audio_note}; {note}")

        return out_image, out_video, " | ".join(notes)


def _run_frames(images: torch.Tensor, params: dict, backend_name: str):
    input_h, input_w = int(images.shape[1]), int(images.shape[2])
    output_w, output_h = common.target_size(input_w, input_h, params["scale"])
    params = dict(params)
    frame_count = int(images.shape[0])
    params["frame_count"] = frame_count
    params["perf_quality"] = common.SCALE_TO_PERF_QUALITY[
        min(common.SCALE_TO_PERF_QUALITY, key=lambda c: abs(c - params["scale"]))
    ]
    backend = _load_backend(backend_name)
    frames = _frame_source(images, params)
    # Preallocate once and fill per frame: accumulating a Python list and then
    # np.stack + .to() would hold several full-size copies of long videos.
    result = torch.empty((frame_count, output_h, output_w, 3), dtype=images.dtype, device=images.device)
    selected_order = None
    progress = _progress_callback()
    for index, output in enumerate(
        backend.process_frames(frames, input_w, input_h, output_w, output_h, params, progress)
    ):
        reference = images[min(index, frame_count - 1)].detach().cpu().numpy()
        corrected, order = _channel_choice(
            output, reference, params["channel_order"] if selected_order is None else selected_order
        )
        if selected_order is None:
            selected_order = order
        corrected = np.clip(corrected, 0.0, 1.0)
        result[index].copy_(torch.from_numpy(corrected))
    return result, input_w, input_h, output_w, output_h, selected_order, params.get("motion_engine", "none")


def _audio_source_path_of(video) -> Optional[str]:
    getter = getattr(video, "get_stream_source", None)
    if not callable(getter):
        return None
    try:
        info = getter()
    except Exception:
        return None
    if isinstance(info, (tuple, list)) and info:
        path = str(info[0])
        return path if os.path.exists(path) else None
    if isinstance(info, str) and os.path.exists(info):
        return info
    return None


NODE_CLASS_MAPPINGS = {
    "RH_DLSS5Enhance": RH_DLSS5Enhance,
    "RH_DLSS5FrameInterpolation": RH_DLSS5FrameInterpolation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_DLSS5Enhance": "RH DLSS5 Enhance (NR / Upscale)",
    "RH_DLSS5FrameInterpolation": "RH DLSS5 Frame Interpolation",
}
