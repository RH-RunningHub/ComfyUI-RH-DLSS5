[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)
[![English](https://img.shields.io/badge/Language-English-2563EB)](./README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-EF4444)](./README_CN.md)

![License](https://img.shields.io/badge/License-MIT-green)

# ComfyUI-RH-DLSS5

Run **NVIDIA DLSS 5 Neural Rendering (NGX feature 18)** over ComfyUI images and videos, with optional 1.5x–3x neural upscaling and DLSS frame generation (frame-rate multiplication). Two nodes, two backends each:

- **Windows 10/11 x64** — the D3D12/NGX bridge (`dlss5nr_bridge.dll`) is loaded in-process via ctypes.
- **Linux** — frames are streamed over pipes to `dlss5nr_host.exe` running under **Wine** (DXVK-NVAPI + vkd3d-proton). The Linux path was verified end-to-end on Ubuntu 22.04 + RTX 4090 (driver 580.95) with Wine 11.0 / DXVK 3.1 / vkd3d-proton 3.0.1.

Outputs are plain ComfyUI types: connect `images` to **SaveImage** and `video` to **SaveVideo**.

## ✨ Features

- Node `RH_DLSS5Enhance` (category **image/upscaling**): DLAA 1x enhance or fixed NVIDIA upsampling factors 1.5x / 1.724x / 2x / 3x.
- Node `RH_DLSS5FrameInterpolation` (category **video**): NVIDIA DLSS frame generation — 2x / 3x / 4x output frame rate with AI-generated frames interleaved between the sources (no frame blending). VIDEO inputs are interpolated and encoded **streaming** (frames go straight into the muxer), so peak host memory stays bounded regardless of clip length; audio is stream-copied from the source when available.
- IMAGE and VIDEO I/O; batch = temporal order, audio preserved where supported.
- Style / intensity / tone / structure / skin tuning parameters mapped to the official NGX DLSSNR properties, with automatic skin-region masking.
- Optical-flow motion vectors (OpenCV / hardware NVOFA) or zero motion for stills; scene-cut history reset.
- Two backends, same node: in-process Windows bridge and Linux/Wine pipe worker; stateless per run and queue-safe with `IS_CHANGED` caching.

### Official native implementation

These nodes do not reimplement or approximate DLSS 5 and do not wrap any HTTP service. They drive the **official NVIDIA NGX runtime** through the same D3D12 contracts a game engine would use:

- `RH_DLSS5Enhance` — `nvngx_dlssnr.dll` (neural rendering feature 18), plus `nvngx_dlss.dll` as the ordinary DLSS Super Resolution carrier for >1x modes:

```
IMAGE / VIDEO frames (render size)
  -> RGB float32 + FP16 motion vectors (optical flow or zero)
  -> D3D12 textures -> NGX feature 18 (feature 1 carrier first when upscaling)
  -> RGB float32 (output size)
  -> IMAGE / VIDEO
```

- `RH_DLSS5FrameInterpolation` — `nvngx_dlssg.dll` (DLSS frame generation) driven by the direct D3D12 NGX host `dlssg-worker.exe` (MIT-licensed, from the DLSS 5 Visual Enhancer lineage):

```
IMAGE / VIDEO frames (source frame rate)
  -> RGBA8 + FP16 motion vectors (DIS / NVOFA, scene-cut aware)
  -> dlssg-worker.exe (D3D12 + NGX DLSS-FG, one generated frame per evaluation)
  -> interleaved CFR timeline (T -> 2T-1 @ 2x, cascaded 3x / 4x grids)
  -> IMAGE / VIDEO
```

Frame-count semantics (verified on RTX 4090, wine 11 / DXVK): evaluating input frame i emits the interpolated frame for the interval (i-1, i); the final interval has no future frame, so a 2x run yields 2T-1 frames at exactly 2x fps. Scene cuts reset history and produce no generated frame for that interval.

For VIDEO inputs the node runs this pipeline as a stream: decoded frames are converted to uint8 once (in bounded chunks, the float batch is released immediately), every output frame is piped directly into an ffmpeg rawvideo muxer (audio stream-copied from the source file), and no full-length float32 output tensor is ever built. 2x holds no intermediate frame buffer at all; 3x/4x keep a single uint8 grid buffer between the cascade stages. IMAGE-batch inputs return the full float tensor (standard ComfyUI `IMAGE`) and use the preallocated in-memory path.

The project-owned parts (bridge, host transport, caller shim, Python nodes) are MIT licensed. The NVIDIA runtime DLLs and the frame-generation worker are user-supplied and **never downloaded or redistributed** by this plugin (see runtime/README.txt for placement).

## Requirements

| Item | Requirement |
| --- | --- |
| GPU | NVIDIA RTX. The tested DLSSNR 310.8 build supports RTX 30/40; stock SDK builds may be RTX 50 only |
| Driver | Current NVIDIA display driver |
| NVIDIA runtime | `nvngx_dlssnr.dll` (always), `nvngx_dlss.dll` (only for >1x), `_nvngx.dll` (see runtime/README.txt); frame interpolation additionally needs the `dlssg/` runtime folder (see Model Download) |
| Windows | Nothing else; the bridge runs in-process |
| Linux | WineHQ stable (11.0 tested) with DXVK-NVAPI + vkd3d-proton DLLs in the prefix (see "Extra Linux setup"); `xvfb` on headless servers (the node manages the virtual X itself); OpenCV for optical flow (optional) |
| ComfyUI | A build with the `VIDEO` type (LoadVideo/SaveVideo) for video I/O; IMAGE works everywhere |

## 🛠️ Installation

1. Clone/copy this folder into `ComfyUI/custom_nodes/ComfyUI-RH-DLSS5`.
2. Install the NVIDIA runtime DLLs — see **📦 Model Download & Installation** below.
3. Linux only: follow "Extra Linux setup" below to prepare the Wine + DXVK stack.
4. Optional: install `opencv-python` for optical-flow motion guides.
5. Restart ComfyUI. The nodes appear as `RH DLSS5 Enhance (NR / Upscale)` (node id `RH_DLSS5Enhance`, category **image/upscaling**) and `RH DLSS5 Frame Interpolation` (node id `RH_DLSS5FrameInterpolation`, category **video**).

## 📦 Model Download & Installation

This plugin drives the official NVIDIA NGX runtime; the three runtime DLLs are **user-supplied** and never downloaded or redistributed by the plugin. Place them (unrenamed) in `ComfyUI/models/dlss5/` — the recommended location, which survives custom_nodes updates. The plugin's bundled `runtime/` folder works as a legacy fallback location.

### Model Directory Structure

```
ComfyUI/
└── models/
    └── dlss5/
        ├── _nvngx.dll              # NGX core loader (~1.4 MB)
        ├── nvngx_dlss.dll          # DLSS Super Resolution carrier (~59 MB), >1x modes only
        ├── nvngx_dlssnr.dll        # DLSSNR neural rendering runtime (~166 MB) — universal build
        ├── nvngx_dlssnr_rtx40.dll  # optional RTX 30/40-lineage build, auto-selected on those GPUs
        └── dlssg/                  # frame interpolation runtime (RH_DLSS5FrameInterpolation):
            ├── dlssg-worker.exe    #   D3D12 NGX frame generation host (MIT, not an NVIDIA binary)
            ├── nvngx.dll           #   NGX SDK loader stub
            ├── _nvngx.dll          #   same file as the copy above
            └── nvngx_dlssg.dll     #   DLSS frame generation snippet (~7.5 MB, user-supplied)
```

### Download Methods (manual — redistribution is not permitted)

| File | Source |
| --- | --- |
| `nvngx_dlssnr.dll` | Universal DLSSNR runtime build covering RTX 20 through 50 (required) |
| `nvngx_dlssnr_rtx40.dll` | Optional dedicated RTX 30/40-lineage 310.8 build; auto-loaded on RTX 30/40 when present |
| `nvngx_dlss.dll` | Official NVIDIA DLSS SDK releases ([github.com/NVIDIA/DLSS/releases](https://github.com/NVIDIA/DLSS/releases), `ngx_dlss_demo_windows.zip`), only needed for >1x |
| `_nvngx.dll` | Extract from an NVIDIA driver package: `7z e <driver>.exe "Display.Driver/_nvngx.dll"` — see `runtime/README.txt` |

### GPU Selection Guide

| GPU | Tested | Notes |
| --- | --- | --- |
| RTX 30 / 40 series | ✅ (RTX 4090, driver 580.95.05) | Prefers `nvngx_dlssnr_rtx40.dll` when present, else the universal build |
| RTX 50 series / RTX 6000D (SM120) | ✅ (RTX 6000D, driver 580.95.05) | Uses the universal `nvngx_dlssnr.dll` |

The plugin picks the runtime build automatically from the GPU's compute capability (Ampere/Ada prefer the `_rtx40` file, every other generation uses the universal one). Set `DLSS5NR_SNR_FILENAME` to a plain file name in the runtime folder to override.

No HuggingFace/ModelScope mirrors exist for these DLLs — any third-party mirror is unofficial and untrusted; download only from NVIDIA.

### Extra Linux setup

Applies to Ubuntu 22.04/24.04 (adapt package names on other distros). Prerequisite: NVIDIA driver **580+** with Vulkan — `nvidia-smi` works and `vulkaninfo --summary` lists the GPU; `nvidia_icd.json` normally ships with the driver.

```bash
# 1) Base dependencies: Xvfb (virtual X for headless servers, required by DXVK WSI),
#    Vulkan loader, zstd
sudo apt update
sudo apt install -y xvfb libvulkan1 vulkan-tools zstd

# 2) Wine 11 (WineHQ stable; Ubuntu 22.04 jammy shown, use noble on 24.04 —
#    and change the file name to winehq-noble.sources)
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
wget -qO- https://dl.winehq.org/wine-builds/winehq.key | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key
sudo wget -qO /etc/apt/sources.list.d/winehq.sources https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources
sudo apt update
sudo apt install -y --install-recommends winehq-stable
wine --version   # expect wine-11.0

# 2b) Fallback: if step 2 fails on i386 dependencies (third-party repos pinning
#     libbrotli1 etc. drag in the whole wine i386 chain), go 64-bit only —
#     download the two debs and unpack without installing; this plugin never
#     uses the 32-bit side:
mkdir -p /tmp/wsx && cd /tmp/wsx
apt-get download wine-stable-amd64 wine-stable
mkdir root && for d in ./*.deb; do dpkg -x "$d" root; done
sudo cp -a root/opt/wine-stable /opt/
ls /opt/wine-stable/share/wine/nls/l_intl.nls   # must exist (ships in the wine-stable package; wineboot fails without it)
# unpacking without dpkg skips the /usr/bin alternatives symlinks — add them:
sudo ln -sf /opt/wine-stable/bin/wine /usr/local/bin/wine
sudo ln -sf /opt/wine-stable/bin/wineboot /usr/local/bin/wineboot
sudo ln -sf /opt/wine-stable/bin/wineserver /usr/local/bin/wineserver
wine --version   # expect wine-11.0

# 3) Initialise the prefix (default ~/.wine; export WINEPREFIX=/path/prefix first to relocate)
wineboot -u

# 4) Copy the 64-bit DXVK 3.1 + vkd3d-proton 3.0.1 + DXVK-NVAPI 0.9.2 DLLs into the prefix system32
mkdir -p /tmp/dlss5-stack && cd /tmp/dlss5-stack
wget -q https://github.com/doitsujin/dxvk/releases/download/v3.1/dxvk-3.1.tar.gz
wget -q https://github.com/HansKristian-Work/vkd3d-proton/releases/download/v3.0.1/vkd3d-proton-3.0.1.tar.zst
wget -q https://github.com/jp7677/dxvk-nvapi/releases/download/v0.9.2/dxvk-nvapi-v0.9.2.tar.gz
tar xzf dxvk-3.1.tar.gz
tar --zstd -xf vkd3d-proton-3.0.1.tar.zst
mkdir nvapi && tar xzf dxvk-nvapi-v0.9.2.tar.gz -C nvapi   # note: this tarball has no top-level dir — extract into a subdir
P="${WINEPREFIX:-$HOME/.wine}/drive_c/windows/system32"
cp dxvk-3.1/x64/dxgi.dll dxvk-3.1/x64/d3d11.dll "$P/"
cp vkd3d-proton-3.0.1/x64/d3d12.dll vkd3d-proton-3.0.1/x64/d3d12core.dll "$P/"
cp nvapi/x64/nvapi64.dll nvapi/x64/nvofapi64.dll "$P/"
```

Layered self-check (if anything fails, locate the broken layer; expected result per line):

```bash
wine --version                     # ① wine-11.0
ls /opt/wine-stable/share/wine/nls/l_intl.nls   # ② exists; ②ok but ③ fails = wineboot dies on l_intl.nls
P="${WINEPREFIX:-$HOME/.wine}/drive_c/windows/system32"
ls "$P" | grep -cE '^(dxgi|d3d11|d3d12|d3d12core|nvapi64|nvofapi64)\.dll$'   # ③ expect 6
Xvfb :150 -screen 0 1280x720x24 >/dev/null 2>&1 & sleep 2
DISPLAY=:150 vulkaninfo --summary 2>/dev/null | grep -m1 deviceName
#    ④ expect NVIDIA GeForce RTX xxxx. Note: without DISPLAY only llvmpipe is listed —
#    that is normal, not a failure. ④ fails while bare vulkaninfo works → driver
#    userspace libs incomplete (typically missing libnvidia-rtcore in containers;
#    vkd3d then fails with vr -3)
pkill -x Xvfb
```

Notes:

- DLL overrides and NVAPI switches are injected by the node automatically (`WINEDLLOVERRIDES=d3d12,d3d12core,nvapi64,dxgi=n,b`, `DXVK_ENABLE_NVAPI=1`, `DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE=on`); the prefix only needs the files in place.
- Xvfb is managed by the node: when `DISPLAY` is unset it ensures one shared virtual X (first free display from `:99`), reuses it across runs, and handles displays left behind by killed workers (adopted if the server still answers, otherwise their stale lock/socket files are removed). Do not wrap with `xvfb-run` (it leaks NGX logs into the binary stdout frame stream).
- If the wine i386 packages cannot be installed (multiarch dependency conflicts), use the 64-bit-only fallback in step 2b above; the test-container recovery script `container_wine_setup.sh` is the unattended version of exactly that.
- Use a proxy if GitHub downloads are slow.

### Linux troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `wineboot` fails with `failed to load l_intl.nls` | `/opt/wine-stable/share/wine` missing (the nls data ships in the wine-stable package — unpack BOTH debs in the 2b fallback) | Re-run self-check ②; re-extract the wine-stable deb and reinstall share |
| vulkaninfo shows no NVIDIA (llvmpipe only) | No DISPLAY, or a zombie Xvfb (process alive, `/tmp/.X11-unix/X*` socket gone). **Without DISPLAY, llvmpipe-only is normal** | `pkill -x Xvfb` and re-run self-check ④; the node manages its own Xvfb |
| vulkaninfo fails with `ERROR_OUT_OF_HOST_MEMORY` | Broken `nvidia_icd.json` (0-byte in container images is common) | `VK_LOADER_DEBUG=error vulkaninfo --summary` will name the JSON parse failure; rewrite the ICD and retest |
| Node fails with `vr -3` / `VK_ERROR_INITIALIZATION_FAILED` | Incomplete driver userspace libs — typically missing `libnvidia-rtcore` in containers (vkd3d enables RT extensions at device creation; plain vkcube never notices) | Mirror all host `580.*` libs into the container and re-point the `libGLX_nvidia.so.0` symlink |
| Node fails with `0xBAD00001` / `Failed to get NvPhysicalGpuHandle for LUID` | Host has **no active display output**: the NVIDIA Linux Vulkan driver then reports `deviceLUIDValid=false`, and dxvk-nvapi ≤0.9.2 answers NvAPI LUID queries with an error (hosts with a connected monitor are unaffected) | Use a dxvk-nvapi build that falls back to the DXGI adapter LUID when `deviceLUIDValid=false` (upstream fix pending; on servers a virtual display/Xorg with a connected output also works) |
| Node fails with `Could not create a D3D12 device` | DISPLAY invalid, or Vulkan is already broken | Pass self-check ④ first; then look for leftover wineserver/Xvfb processes |
| Task hangs with no output, and retries hang too | Orphaned `dlss5nr_host.exe` / `wineserver` from a previously killed run holding the GPU/pipes | `pkill -f 'dlss5nr_hos[t]'; pkill -x wineserver; pkill -x Xvfb`, then retry |

### Environment variables

| Variable | Meaning |
| --- | --- |
| `DLSS5_RUNTIME_DIR` | Runtime folder override (`RH_DLSS5Enhance`) |
| `DLSS5_FG_RUNTIME_DIR` | Frame-generation runtime folder override (`RH_DLSS5FrameInterpolation`) |
| `DLSS5_GPU_INDEX` | NVIDIA adapter index for the worker (default 0). Deliberately an environment variable, not a widget: on a queue platform each worker is pinned to one GPU, and a per-task widget value would disagree with the scheduling |
| `DLSS5NR_SNR_FILENAME` | Plain file name of an alternative `nvngx_dlssnr*.dll` in the runtime folder (auto-selection override) |
| `DLSS5_WINE` | Wine binary override (Linux) |
| `DLSS5_WINEPREFIX` / `WINEPREFIX` | Wine prefix for the worker |

## 📝 Node Reference

### RH_DLSS5Enhance (NR / upscale)

Connect **either** `image` or `video` (or both).

| Input | Type | Notes |
| --- | --- | --- |
| `image` | IMAGE | Batch = temporal order |
| `video` | VIDEO | From LoadVideo; frames processed in playback order |
| `upscaling_mode` | combo | `1x (DLAA)` native enhance, `1.5x / 1.724x / 2x / 3x` fixed NVIDIA factors (carrier + feature 18) |
| `style` | combo | default / natural / cinematic, or **off (bypass NR)** — disables all processing, frames pass through untouched |
| `preset` | INT 0–9, default 0 | Internal render preset hint, passed to `DLSSNR.Hint.Render.Preset`. Keep 0 unless the docs of your runtime build define another value |
| `intensity` | FLOAT 0.0–2.0, default 1.0 | Overall neural pass strength (`DLSSNR.Intensity`). 1.0 is full strength; >1.0 usually has no extra effect; <1.0 blends back towards the source for lighter denoising |
| `tone` | FLOAT 0.0–2.0, default 1.0 | Local tone mapping strength (`DLSSNR.LocalToneStrength`; the global term `GlobalToneStrength` is fixed at 1.0). Higher = punchier local contrast, lower = flatter |
| `structure` | FLOAT 0.0–2.0, default 1.5 | Local detail / structure reconstruction strength (`DLSSNR.LocalStructureStrength`). Higher = sharper with more texture; lower = softer and cleaner (drop it if the result looks plasticky) |
| `skin` | FLOAT −1.0–2.0, default 2.0 | Skin structure reconstruction strength (`DLSSNR.SkinStructureStrength`). **Only active with `auto_mask=on`**; −1 leaves the decision to the model |
| `auto_mask` | combo on/off, default on | Automatic skin-region detection (`DLSSNR.UseAutoMask`): the model finds skin itself and applies `skin` there — effectively the on/off switch for `skin` |
| `motion` | combo | auto (optical flow when OpenCV present) / optical_flow / none. Implausible flow fields (fast pans, noise) are treated as scene cuts so temporal history cannot accumulate ghosting |
| `scene_change_threshold` | FLOAT | Scene-cut history reset threshold (optical flow only) |
| `batch_mode` | combo | `temporal sequence` keeps history between frames; `still images` resets every frame |
| `warmup_frames` | INT | Warm-up budget reported to the worker (0 is fine) |
| `keep_audio` | combo | Keep original audio in the VIDEO output when supported |
| `backend` | combo | auto / linux-wine / windows-bridge |
| `channel_order` | combo | auto-detects R/B swap of the runtime output; force RGBA/BGRA on wrong colours |
| `runtime_dir` | STRING | Optional runtime folder override; when empty the lookup order is `DLSS5_RUNTIME_DIR` → `<ComfyUI>/models/dlss5` → the bundled `runtime/` |
| `wine_prefix` | STRING | Linux: WINEPREFIX override for the worker |

| Output | Type | Meaning |
| --- | --- | --- |
| `images` | IMAGE | Processed frames (set when `image` was connected) |
| `video` | VIDEO | Processed video object for SaveVideo (set when `video` was connected; on old ComfyUI builds without the VIDEO type this is None and the `status` output names the written file) |
| `status` | STRING | One-line summary: frames, sizes, backend, channel order, output notes |

### RH_DLSS5FrameInterpolation (frame-rate multiplication)

Connect **either** `video` (frame rate and audio are taken from it) or `image`. For VIDEO inputs the output video is encoded streaming and `images` is None; for IMAGE inputs `video` is an in-memory VIDEO object and `images` carries the full batch.

| Input | Type | Notes |
| --- | --- | --- |
| `video` | VIDEO | From LoadVideo; source frame rate and audio come from this input |
| `image` | IMAGE | Batch = frames in temporal order; batches are treated as a 24fps source |
| `output_fps` | combo | `2x` / `3x` / `4x` frame-rate multiplication, or an exact target output rate (23.976–144 fps presets). For a target rate the node builds the smallest 2x/4x/8x dense grid and nearest-picks the target timeline (no duplicated frames); target rates must exceed the source rate and stay within 6x |
| `motion` | combo | `auto` = hardware NVOFA optical flow when available, else OpenCV DIS; `nvof` / `dis` force one guide. Generated frames are skipped across scene cuts |
| `scene_change_threshold` | FLOAT 0.01–1.0, default 0.24 | Mean luminance change above which temporal history resets (scene-cut detection). Higher = fewer resets |
| `keep_audio` | combo on/off, default on | Keep the original audio in the VIDEO output: stream-copied from the source file, re-encoded from the AUDIO input when no file is available |
| `audio` | AUDIO | Optional audio socket: replaces the source audio on VIDEO inputs (while keep_audio is on) and is the only way to attach sound to IMAGE inputs. Re-encoded to AAC when it cannot be stream-copied |
| `runtime_dir` | STRING | Optional override of the folder holding `dlssg-worker.exe + nvngx.dll + _nvngx.dll + nvngx_dlssg.dll`; lookup order `DLSS5_FG_RUNTIME_DIR` → `<models>/dlss5/dlssg` → bundled `runtime/dlssg` |
| `wine_prefix` | STRING | Linux: WINEPREFIX override for the worker (needs DXVK + vkd3d-proton + DXVK-NVAPI, same stack as the enhance node) |

| Output | Type | Meaning |
| --- | --- | --- |
| `images` | IMAGE | Interleaved frames (IMAGE input only; None for VIDEO input) |
| `video` | VIDEO | Interpolated video for SaveVideo (streaming-encoded temp file for VIDEO input) |
| `status` | STRING | One-line summary: guide engine, source→target fps, frame counts, runtime path |

## 🚀 Usage

How to tune the neural rendering params (`style` is an independent overall style profile: default/natural/cinematic):

- **Video denoising / compression-artifact cleanup**: keep `intensity` at 1.0; raise `structure` if detail gets wiped, lower it if noise shows through;
- **Portraits**: `auto_mask=on` with `skin` at its default 2.0 — smooth skin that keeps structure; non-skin areas are unaffected;
- **Result looks over-processed**: lower `structure` first, then `intensity` if needed; `tone` usually stays alone.

### Example Workflow

- [`examples/ComfyUI-RH-DLSS5_image_api.json`](examples/ComfyUI-RH-DLSS5_image_api.json) — LoadImage → RH_DLSS5Enhance → SaveImage.
- [`examples/ComfyUI-RH-DLSS5_video_api.json`](examples/ComfyUI-RH-DLSS5_video_api.json) — LoadVideo → RH_DLSS5Enhance → SaveVideo (needs a recent ComfyUI build for LoadVideo/SaveVideo).
- [`examples/ComfyUI-RH-DLSS5_frame_interp_api.json`](examples/ComfyUI-RH-DLSS5_frame_interp_api.json) — LoadVideo → RH_DLSS5FrameInterpolation (2x) → SaveVideo.

Import the JSON into ComfyUI and replace the placeholder media names in the Load nodes before submitting.

Frame interpolation tips: 24→48 fps with `2x` is the sweet spot (every interval gets exactly one generated frame); `3x`/`4x` are built from cascaded 2x passes and cost proportionally more time and memory. On long clips the VIDEO path encodes while interpolating, so memory grows with the decoder, not the clip length.

## Queueing and caching

The node is stateless per run and safe for queued prompts: Linux spawns one Wine worker per execution (serialized by an internal lock), Windows keeps one bridge session per GPU and reuses it when sizes/settings match. `IS_CHANGED` hashes parameters plus a probe of the input tensor / source video metadata, so re-queued identical work is served from ComfyUI's cache.

## Verification

Verified end-to-end on the Linux path: 320x240 48-frame test clip, 1x and 1.5x (480x360), Wine 11.0 + DXVK 3.1 + vkd3d-proton 3.0.1 + DXVK-NVAPI on RTX 4090 / driver 580.95.05, runtime DLSSNR 310.8. In 2026-09 a from-scratch install following this document (including the 2b fallback and the layered self-check) was validated on a fresh Ubuntu 22.04 container. The Windows in-process bridge follows the same protocol as the MIT-licensed reference integration.

Frame interpolation verified on the same machine: 2x/3x/4x frame counts and source-slot alignment checked against synthetic moving-content clips (hardware NVOFA and DIS guides, audio stream-copy), and a 400-frame 1080p 2x run completes with roughly 20 GB host RAM peak where the pre-streaming implementation was killed by the memory cgroup on the same clip.

## 🔗 Links

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)

- Upstream reference (Linux/Wine CLI this plugin is based on): <https://github.com/kos94ok/ComfyUI-DLSS5-NR-Linux>
- Upstream reference (Windows in-process bridge): <https://github.com/lisitskyaa/ComfyUI-DLSS5-NR>
- Video-worker reference (Merserk protocol): <https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer>
- Frame-generation worker & scheduler reference: <https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation>
- Official NVIDIA DLSS SDK (source of `nvngx_dlss.dll`): <https://github.com/NVIDIA/DLSS/releases>
- NVIDIA DLSS repository: <https://github.com/NVIDIA/DLSS>
- DXVK: <https://github.com/doitsujin/dxvk> · vkd3d-proton: <https://github.com/HansKristian-Work/vkd3d-proton> · DXVK-NVAPI: <https://github.com/jp7677/dxvk-nvapi>

## 📄 License

Plugin code and project-owned native components: **MIT** — see [LICENSE](LICENSE). The native bridge/host sources under `native/src/` derive from the MIT-licensed ComfyUI-DLSS5-NR projects listed above.

NVIDIA NGX/DLSS runtime binaries (`nvngx_dlssnr.dll`, `nvngx_dlss.dll`, `_nvngx.dll`) are **proprietary NVIDIA software**: they are not included, not downloaded, and not redistributed. Use only builds you are legally entitled to use. DLSS and NGX are NVIDIA trademarks.

## 🙏 Acknowledgements

This project is based on the MIT-licensed reference integrations [ComfyUI-DLSS5-NR-Linux](https://github.com/kos94ok/ComfyUI-DLSS5-NR-Linux) (Linux/Wine CLI, whose `dlss5nr_video.py` transport protocol the Linux worker adapts) and [ComfyUI-DLSS5-NR](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR) (Windows in-process bridge). The frame-generation pipeline follows the Merserk-lineage worker protocol and the scheduling design of [ComfyUI-NVIDIA-DLSS-Frame-Interpolation](https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation). It builds on [DXVK](https://github.com/doitsujin/dxvk), [vkd3d-proton](https://github.com/HansKristian-Work/vkd3d-proton) and [DXVK-NVAPI](https://github.com/jp7677/dxvk-nvapi). Thanks to NVIDIA for the NGX/DLSS SDK, and to [@tetsuoo-online](https://github.com/tetsuoo-online) for reporting and pinpointing the Windows backend issue (#1), and to [@MakkiShizu](https://github.com/MakkiShizu) for the Windows optical-flow and worker-launch fixes (#2).
