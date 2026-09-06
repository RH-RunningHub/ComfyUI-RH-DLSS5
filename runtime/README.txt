DLSS5 runtime NVIDIA files (user-supplied — never redistributed)

RECOMMENDED LOCATION: <ComfyUI>/models/dlss5/

Place the three NVIDIA DLLs there (the folder is scanned automatically and
survives custom_nodes updates/re-syncs — unlike this bundled runtime/ folder,
which is kept only as a legacy fallback). The project caller shim does NOT go
there; the plugin links its bundled caller/ into the runtime folder on its own.

Files to place there:

  nvngx_dlssnr.dll   DLSS 5 Neural Rendering runtime (NGX feature 18).
                     Required. Use a build that matches your GPU (the tested
                     310.8 build supports RTX 30/40; stock NVIDIA SDK builds
                     may be RTX 50 only).

  nvngx_dlss.dll     Ordinary DLSS Super Resolution carrier (NGX feature 1).
                     Required only for upscaling above 1x. Official source:
                     https://github.com/NVIDIA/DLSS/releases (ngx_dlss_demo zips).

  _nvngx.dll         NGX core loader. Optional explicit override; on Windows
                     the bridge also searches normal DLL paths and the NVIDIA
                     DriverStore. On Linux/Wine you must copy it there from an
                     NVIDIA Windows driver package:
                       7z e <driver>.exe "Display.Driver/_nvngx.dll" -o<there>

  caller/nvngx.dll_comfy.dll   Project caller shim (shipped with this plugin;
                               only needed when using this legacy runtime/
                               folder instead of models/dlss5).

FRAME INTERPOLATION RUNTIME (RH_DLSS5FrameInterpolation node)

RECOMMENDED LOCATION: <ComfyUI>/models/dlss5/dlssg/

Place these four files there (the folder is scanned automatically):

  dlssg-worker.exe   Direct D3D12 NGX frame generation host. Not an NVIDIA
                     binary: it comes from the MIT-licensed DLSS 5 Visual
                     Enhancer lineage (Merserk / Konohamaru04). 66 KB.
  nvngx.dll          NVIDIA NGX SDK loader stub (ships next to the worker).
  _nvngx.dll         NGX driver core - the same file models/dlss5 already has.
  nvngx_dlssg.dll    NVIDIA DLSS frame generation snippet (NVIDIA App /
                     NGX SDK 310.7 runtime, ~7.5 MB). User-supplied like the
                     other NVIDIA DLLs; never redistributed.

DLSS 5 and the NGX runtime are proprietary NVIDIA software. Use only builds
you are legally entitled to use.
