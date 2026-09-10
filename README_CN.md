[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)
[![English](https://img.shields.io/badge/Language-English-2563EB)](./README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-EF4444)](./README_CN.md)

![License](https://img.shields.io/badge/License-MIT-green)

# ComfyUI-RH-DLSS5

在 ComfyUI 里对图像和视频运行 **NVIDIA DLSS 5 神经渲染（NGX feature 18）**，支持 1.5x–3x 神经放大和 DLSS 帧生成（帧率倍增）。两个节点、各自双后端：

- **Windows 10/11 x64** — 通过 ctypes 在 ComfyUI 进程内加载 D3D12/NGX 桥（`dlss5nr_bridge.dll`）。
- **Linux** — 帧经管道流式传给 **Wine** 下运行的 `dlss5nr_host.exe`（DXVK-NVAPI + vkd3d-proton）。Linux 路线已在 Ubuntu 22.04 + RTX 4090（驱动 580.95）+ Wine 11.0 / DXVK 3.1 / vkd3d-proton 3.0.1 上端到端验证通过。

输出是 ComfyUI 标准类型：`images` 接 **SaveImage**，`video` 接 **SaveVideo**。

## ✨ 功能特点

- 节点 `RH_DLSS5Enhance`（**image/upscaling** 分类）：DLAA 1x 原尺寸增强，或 NVIDIA 固定档 1.5x / 1.724x / 2x / 3x 放大。
- 节点 `RH_DLSS5FrameInterpolation`（**video** 分类）：NVIDIA DLSS 帧生成——可按目标输出帧率选择（也兼容 2x / 3x / 4x 倍率），AI 生成的插值帧直接插入源帧之间（不是融合混帧）。VIDEO 输入全程**流式**处理：插帧结果直接送编码器，峰值内存与片长无关；源文件音轨可直接 stream copy。
- IMAGE 与 VIDEO 输入输出；批次顺序即时间顺序，支持尽量保留音轨。
- style / intensity / tone / structure / skin 调参直接映射官方 NGX DLSSNR 属性，支持皮肤区域自动遮罩。
- 光流运动向量（OpenCV / NVOFA 硬件光流）或静帧零向量；镜头切换自动重置时序历史。
- 双后端同一节点：Windows 进程内桥、Linux/Wine 管道 worker；单次执行无状态、队列安全并带 `IS_CHANGED` 缓存。

### 官方库原生实现

本插件不重新实现、不近似 DLSS 5，也不包装任何 HTTP 服务，而是按游戏引擎相同的 D3D12 契约驱动 **NVIDIA 官方 NGX 运行时**：

- `RH_DLSS5Enhance`——`nvngx_dlssnr.dll` 神经渲染 feature 18；>1x 模式先用 `nvngx_dlss.dll` 普通超分载体 feature 1：

```
IMAGE / VIDEO 帧（渲染尺寸）
  -> RGB float32 + FP16 运动向量（光流或零向量）
  -> D3D12 纹理 -> NGX feature 18（放大时先过 feature 1 载体）
  -> RGB float32（输出尺寸）
  -> IMAGE / VIDEO
```

- `RH_DLSS5FrameInterpolation`——`nvngx_dlssg.dll`（DLSS 帧生成），由直连 D3D12 的 NGX 宿主 `dlssg-worker.exe` 驱动（MIT 协议，出自 DLSS 5 Visual Enhancer 一系）：

```
IMAGE / VIDEO 帧（源帧率）
  -> RGBA8 + FP16 运动向量（DIS / NVOFA，镜头切换感知）
  -> dlssg-worker.exe（D3D12 + NGX DLSS-FG，每次评估生成一帧）
  -> 交错 CFR 时间线（2x 时 T -> 2T-1 帧，3x/4x 走级联网格）
  -> IMAGE / VIDEO
```

帧数语义（RTX 4090、wine 11 / DXVK 实测）：worker 处理输入第 i 帧时生成 (i-1, i) 区间的插值帧；最后一段区间没有"未来帧"可插，因此 2x 输出恰为 2T-1 帧、帧率恰为 2 倍。镜头切换处重置历史、该区间不产出生成帧。

VIDEO 输入时整条管线以流式运行：解码出的帧分块转成 uint8（随即释放 float 批次），每个输出帧直接送进 ffmpeg rawvideo 封装器（音轨从源文件 stream copy），全程不物化全片 float32 输出张量。2x 完全没有中间帧缓冲；3x/4x 级联级间只保留一个 uint8 网格缓冲。IMAGE 批次输入走预分配内存路径，返回标准 float 张量。

项目自有部分（桥、host 传输、caller shim、Python 节点）为 MIT 协议。NVIDIA 运行时 DLL 与帧生成 worker 由用户自行提供，本插件**不下载、不再分发**（放置位置见 runtime/README.txt）。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| GPU | NVIDIA RTX。实测 DLSSNR 310.8 构建支持 RTX 30/40；官方 SDK 原生构建可能仅 RTX 50 |
| 驱动 | 较新的 NVIDIA 显示驱动 |
| NVIDIA 运行时 | `nvngx_dlssnr.dll`（必须）、`nvngx_dlss.dll`（仅 >1x 需要）、`_nvngx.dll`（见 runtime/README.txt）；帧插值另需 `dlssg/` 运行时目录（见模型下载章节） |
| Windows | 无其他要求，桥进程内运行 |
| Linux | WineHQ stable（实测 11.0），prefix 内放置 DXVK + vkd3d-proton + DXVK-NVAPI 的 64 位 DLL（按下节「Linux 额外安装」）；无桌面服务器装 `xvfb`（虚拟 X 由节点自管）；可选 OpenCV 用于光流 |
| ComfyUI | 视频输入输出需要带 `VIDEO` 类型的较新构建（LoadVideo/SaveVideo）；IMAGE 全版本可用 |

## 🛠️ 安装

1. 将本目录克隆/复制到 `ComfyUI/custom_nodes/ComfyUI-RH-DLSS5`。
2. 安装 NVIDIA 运行时 DLL——见下方 **📦 模型下载与安装**。
3. 仅 Linux：按下节「Linux 额外安装」准备 Wine + DXVK 栈。
4. 可选：安装 `opencv-python` 启用光流运动向量。
5. 重启 ComfyUI。节点有两个：`RH DLSS5 Enhance (NR / Upscale)`（节点 id `RH_DLSS5Enhance`，**image/upscaling** 分类）和 `RH DLSS5 Frame Interpolation`（节点 id `RH_DLSS5FrameInterpolation`，**video** 分类）。

## 📦 模型下载与安装

本插件驱动 NVIDIA 官方 NGX 运行时，三个运行时 DLL 由**用户自行提供**，插件不下载、不再分发。请不改名放入 `ComfyUI/models/dlss5/`——推荐位置，插件更新/重同步不会丢；插件自带的 `runtime/` 目录仅作旧版兜底位置。

### 目录结构

```
ComfyUI/
└── models/
    └── dlss5/
        ├── _nvngx.dll              # NGX core 加载器（~1.4 MB）
        ├── nvngx_dlss.dll          # DLSS 超分载体（~59 MB），仅 >1x 模式需要
        ├── nvngx_dlssnr.dll        # DLSSNR 神经渲染运行时（~166 MB）——通用构建
        ├── nvngx_dlssnr_rtx40.dll  # 可选的 RTX 30/40 系专用构建，对应显卡自动加载
        └── dlssg/                  # 帧插值节点运行时（RH_DLSS5FrameInterpolation）：
            ├── dlssg-worker.exe    #   D3D12 NGX 帧生成宿主（MIT，非 NVIDIA 二进制）
            ├── nvngx.dll           #   NGX SDK 加载桩
            ├── _nvngx.dll          #   与上目录同名文件相同
            └── nvngx_dlssg.dll     #   DLSS 帧生成片段（~7.5 MB，用户自备）
```

### 下载方式（手动——不允许再分发）

| 文件 | 来源 |
| --- | --- |
| `nvngx_dlssnr.dll` | 覆盖 RTX 20–50 的通用 DLSSNR 运行时构建（必须） |
| `nvngx_dlssnr_rtx40.dll` | 可选的 RTX 30/40 系 310.8 专用构建；存在时 RTX 30/40 自动优先加载 |
| `nvngx_dlss.dll` | NVIDIA 官方 DLSS SDK releases（[github.com/NVIDIA/DLSS/releases](https://github.com/NVIDIA/DLSS/releases)，`ngx_dlss_demo_windows.zip`），仅 >1x 需要 |
| `_nvngx.dll` | NVIDIA 驱动包提取：`7z e <driver>.exe "Display.Driver/_nvngx.dll"`——详见 `runtime/README.txt` |

### 显卡选择指南

| GPU | 实测 | 说明 |
| --- | --- | --- |
| RTX 30 / 40 系列 | ✅（RTX 4090，驱动 580.95.05） | 优先加载 `nvngx_dlssnr_rtx40.dll`，没有则回退通用构建 |
| RTX 50 系列 / RTX 6000D（SM120） | ✅（RTX 6000D，驱动 580.95.05） | 使用通用 `nvngx_dlssnr.dll` |

插件按显卡算力自动选择运行时构建（Ampere/Ada 优先 `_rtx40` 文件，其余代次用通用构建）。如需手动指定，把 `DLSS5NR_SNR_FILENAME` 设为运行时目录里的纯文件名即可。

这些 DLL 没有 HuggingFace/ModelScope 官方镜像——任何第三方镜像都非官方且不可信，请只从 NVIDIA 官方渠道获取。

### Linux 额外安装

适用 Ubuntu 22.04/24.04（其它发行版包名等价替换）。前提：NVIDIA 驱动 **580+** 且带 Vulkan——`nvidia-smi` 正常、`vulkaninfo --summary` 能列出显卡，`nvidia_icd.json` 一般随驱动自带。

```bash
# 1) 基础依赖：Xvfb（无桌面服务器的虚拟 X，DXVK 的 WSI 必需）、Vulkan loader、zstd
sudo apt update
sudo apt install -y xvfb libvulkan1 vulkan-tools zstd

# 2) Wine 11（WineHQ stable；Ubuntu 22.04 jammy 为例，24.04 换成 noble 并把文件名改为 winehq-noble.sources）
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
wget -qO- https://dl.winehq.org/wine-builds/winehq.key | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key
sudo wget -qO /etc/apt/sources.list.d/winehq.sources https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources
sudo apt update
sudo apt install -y --install-recommends winehq-stable
wine --version   # 期望 wine-11.0

# 2b) 备选：第 2 步因 i386 依赖冲突装不上时（第三方源顶掉 libbrotli1 等会连坐整个
#     wine i386 链），改纯 64 位方案——下载两个 deb 只解包不安装，32 位侧本插件用不到：
mkdir -p /tmp/wsx && cd /tmp/wsx
apt-get download wine-stable-amd64 wine-stable
mkdir root && for d in ./*.deb; do dpkg -x "$d" root; done
sudo cp -a root/opt/wine-stable /opt/
ls /opt/wine-stable/share/wine/nls/l_intl.nls   # 必须存在（在 wine-stable 包里，缺了 wineboot 会失败）
# 解包方式不经过 dpkg，不会自动建 /usr/bin 软链，手动补上：
sudo ln -sf /opt/wine-stable/bin/wine /usr/local/bin/wine
sudo ln -sf /opt/wine-stable/bin/wineboot /usr/local/bin/wineboot
sudo ln -sf /opt/wine-stable/bin/wineserver /usr/local/bin/wineserver
wine --version   # 期望 wine-11.0

# 3) 初始化 prefix（默认 ~/.wine；想放别处就 export WINEPREFIX=/path/prefix 再执行）
wineboot -u

# 4) DXVK 3.1 + vkd3d-proton 3.0.1 + DXVK-NVAPI 0.9.2 的 64 位 DLL 拷进 prefix system32
mkdir -p /tmp/dlss5-stack && cd /tmp/dlss5-stack
wget -q https://github.com/doitsujin/dxvk/releases/download/v3.1/dxvk-3.1.tar.gz
wget -q https://github.com/HansKristian-Work/vkd3d-proton/releases/download/v3.0.1/vkd3d-proton-3.0.1.tar.zst
wget -q https://github.com/jp7677/dxvk-nvapi/releases/download/v0.9.2/dxvk-nvapi-v0.9.2.tar.gz
tar xzf dxvk-3.1.tar.gz
tar --zstd -xf vkd3d-proton-3.0.1.tar.zst
mkdir nvapi && tar xzf dxvk-nvapi-v0.9.2.tar.gz -C nvapi   # 注意：这个包没有顶层目录，必须解进子目录
P="${WINEPREFIX:-$HOME/.wine}/drive_c/windows/system32"
cp dxvk-3.1/x64/dxgi.dll dxvk-3.1/x64/d3d11.dll "$P/"
cp vkd3d-proton-3.0.1/x64/d3d12.dll vkd3d-proton-3.0.1/x64/d3d12core.dll "$P/"
cp nvapi/x64/nvapi64.dll nvapi/x64/nvofapi64.dll "$P/"
```

分层自检（装完任一步出问题，按层定位；每条给出预期结果）：

```bash
wine --version                     # ① wine-11.0
ls /opt/wine-stable/share/wine/nls/l_intl.nls   # ② 存在；②过③败=wineboot 报 l_intl.nls
P="${WINEPREFIX:-$HOME/.wine}/drive_c/windows/system32"
ls "$P" | grep -cE '^(dxgi|d3d11|d3d12|d3d12core|nvapi64|nvofapi64)\.dll$'   # ③ 期望 6
Xvfb :150 -screen 0 1280x720x24 >/dev/null 2>&1 & sleep 2
DISPLAY=:150 vulkaninfo --summary 2>/dev/null | grep -m1 deviceName
#    ④ 期望 NVIDIA GeForce RTX xxxx。注意：不带 DISPLAY 时只列 llvmpipe 属正常现象，别误判；
#    ④失败但裸 vulkaninfo 正常 → 驱动用户态库不齐（典型：容器里缺 libnvidia-rtcore，vkd3d 报 vr -3）
pkill -x Xvfb
```

说明：

- DLL 覆盖与 NVAPI 开关注入由节点自动完成（`WINEDLLOVERRIDES=d3d12,d3d12core,nvapi64,dxgi=n,b`、`DXVK_ENABLE_NVAPI=1`、`DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE=on`），prefix 里只需文件就位。
- Xvfb 由节点自动管理：`DISPLAY` 未设置时确保一个共享虚拟 X（从 `:99` 起找空闲 display），多次运行复用同一个；进程被杀残留的 display 若服务仍在则直接收编，否则清掉遗留的 lock/socket；不要用 `xvfb-run` 包装（会把 NGX 日志窜进二进制 stdout 污染帧流）。
- i386 装不上时用上面 2b) 的纯 64 位方案；测试容器的一键恢复脚本 `container_wine_setup.sh` 就是这个方案的无人值守版。
- GitHub 直连慢时走代理下载。

### Linux 故障排查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `wineboot` 报 `failed to load l_intl.nls` | `/opt/wine-stable/share/wine` 缺失（nls 数据在 wine-stable 包里，2b 解包时两个 deb 都要解） | 重跑自检②；补解 wine-stable deb 后重装 share |
| vulkaninfo 找不到 NVIDIA（只列 llvmpipe） | 无 DISPLAY，或 Xvfb 僵死（进程在、`/tmp/.X11-unix/X*` socket 没了）。**不带 DISPLAY 只列 llvmpipe 属正常** | `pkill -x Xvfb` 后按自检④重测；节点会自管 Xvfb |
| vulkaninfo 报 `ERROR_OUT_OF_HOST_MEMORY` | `nvidia_icd.json` 损坏（容器镜像常见 0 字节） | `VK_LOADER_DEBUG=error vulkaninfo --summary` 会直接报 JSON 解析失败；重写 ICD 后复测 |
| 节点报 `vr -3` / `VK_ERROR_INITIALIZATION_FAILED` | 驱动用户态库不齐——典型是容器缺 `libnvidia-rtcore`（vkd3d 建设备要开 RT 扩展，普通 vkcube 测不出来） | 把宿主机全部 `580.*` 库镜像进容器，重指 `libGLX_nvidia.so.0` 软链 |
| 节点报 `0xBAD00001` / `Failed to get NvPhysicalGpuHandle for LUID` | 宿主机**没有活动显示输出**：此时 NVIDIA Linux Vulkan 驱动上报 `deviceLUIDValid=false`，dxvk-nvapi ≤0.9.2 对 NvAPI 的 LUID 查询直接报错（接了显示器的宿主不受影响） | 换用带「`deviceLUIDValid=false` 时回退 DXGI 适配器 LUID」修复的 dxvk-nvapi 构建（上游修复发布中；服务器上接虚拟显示/带输出的 Xorg 也可绕过） |
| 节点报 `Could not create a D3D12 device for NVIDIA GPU` | DISPLAY 无效或 Vulkan 层面就没通 | 先过自检④；再查 wineserver/Xvfb 残留 |
| 任务挂起无输出，重试也挂 | 上次异常退出留下孤儿 `dlss5nr_host.exe` / `wineserver`，占着 GPU/管道 | `pkill -f 'dlss5nr_hos[t]'; pkill -x wineserver; pkill -x Xvfb` 后重试 |

### 环境变量

| 变量 | 含义 |
| --- | --- |
| `DLSS5_RUNTIME_DIR` | runtime 目录覆盖（`RH_DLSS5Enhance`） |
| `DLSS5_FG_RUNTIME_DIR` | 帧生成 runtime 目录覆盖（`RH_DLSS5FrameInterpolation`） |
| `DLSS5_GPU_INDEX` | worker 使用的 NVIDIA 适配器序号（默认 0）。刻意做成环境变量而非控件：队列平台每台 worker 本就绑定一块卡，控件里的手选值会和调度结果打架 |
| `DLSS5NR_SNR_FILENAME` | 指定运行时目录里备用的 `nvngx_dlssnr*.dll` 纯文件名（覆盖自动选择） |
| `DLSS5_WINE` | Wine 可执行文件覆盖（Linux） |
| `DLSS5_WINEPREFIX` / `WINEPREFIX` | worker 的 Wine prefix |

## 📝 节点参考

### RH_DLSS5Enhance（神经渲染 / 放大）

`image` 和 `video` 至少接一个（也可都接）。

| 输入 | 类型 | 说明 |
| --- | --- | --- |
| `image` | IMAGE | 批次顺序即时间顺序 |
| `video` | VIDEO | 来自 LoadVideo，按播放顺序逐帧处理 |
| `upscaling_mode` | 下拉 | `1x (DLAA)` 原尺寸增强；`1.5x / 1.724x / 2x / 3x` 为 NVIDIA 固定档（载体 + feature 18 两阶段） |
| `style` | 下拉 | default / natural / cinematic,或 **off (bypass NR)**——跳过全部处理,帧原样直通 |
| `preset` | INT，0–9，默认 0 | 内部渲染预设提示，直传 `DLSSNR.Hint.Render.Preset`。保持 0 即可，除非你所用 runtime 的文档明确给出其它值的含义 |
| `intensity` | FLOAT，0.0–2.0，默认 1.0 | 神经渲染整体强度（`DLSSNR.Intensity`）。1.0 为全量；>1.0 通常无额外效果；<1.0 向原图回混，用于轻降噪 |
| `tone` | FLOAT，0.0–2.0，默认 1.0 | 局部影调/对比映射强度（`DLSSNR.LocalToneStrength`；全局影调项 `GlobalToneStrength` 固定 1.0）。调高局部对比更强，调低更平 |
| `structure` | FLOAT，0.0–2.0，默认 1.5 | 局部细节/结构重建强度（`DLSSNR.LocalStructureStrength`）。调高更锐、纹理更多；调低更柔更干净（画面发"塑料感"时降它） |
| `skin` | FLOAT，−1.0–2.0，默认 2.0 | 皮肤区域结构重建强度（`DLSSNR.SkinStructureStrength`）。**仅 `auto_mask=on` 时生效**；设 −1 表示交给模型自行决定 |
| `auto_mask` | 下拉 on/off，默认 on | 皮肤区域自动检测（`DLSSNR.UseAutoMask`）：模型自找皮肤并按 `skin` 参数处理，相当于 `skin` 的总开关 |
| `motion` | 下拉 | auto（装了 OpenCV 就用光流）/ optical_flow / none。不可信的光流场（高速平移、噪声）按镜头切换处理,避免时序历史累积拖影 |
| `scene_change_threshold` | FLOAT | 镜头切换时重置时序历史的阈值（仅光流模式） |
| `batch_mode` | 下拉 | `temporal sequence` 帧间保留时序历史；`still images` 每帧重置 |
| `warmup_frames` | INT | 告知 worker 的预热预算（0 即可） |
| `keep_audio` | 下拉 | 在 VIDEO 输出中尽量保留原音轨 |
| `backend` | 下拉 | auto / linux-wine / windows-bridge |
| `channel_order` | 下拉 | 自动检测运行时输出的 R/B 交换；偏色时强制 RGBA/BGRA |
| `runtime_dir` | 字符串 | 覆盖 runtime 目录；留空时按 `DLSS5_RUNTIME_DIR` → `<ComfyUI>/models/dlss5` → 插件自带 `runtime/` 顺序查找 |
| `wine_prefix` | 字符串 | Linux：覆盖 worker 使用的 WINEPREFIX |

| 输出 | 类型 | 含义 |
| --- | --- | --- |
| `images` | IMAGE | 处理后的帧（接了 `image` 输入时有效） |
| `video` | VIDEO | 交给 SaveVideo 的视频对象（接了 `video` 输入时有效；老版本 ComfyUI 无 VIDEO 类型时为 None，输出文件路径写在 `status` 里） |
| `status` | STRING | 单行摘要：帧数、尺寸、后端、通道顺序、输出说明 |

### RH_DLSS5FrameInterpolation（帧率倍增）

`video`（源帧率与音轨来自它）或 `image` 至少接一个。VIDEO 输入的输出视频为流式编码、`images` 为 None；IMAGE 输入则 `video` 是内存 VIDEO 对象、`images` 返回完整批次。

| 输入 | 类型 | 说明 |
| --- | --- | --- |
| `video` | VIDEO | 来自 LoadVideo；源帧率与音轨取自该输入 |
| `image` | IMAGE | 批次顺序即时间顺序；批次按 24fps 源处理 |
| `output_fps` | 下拉 | `2x` / `3x` / `4x` 倍率插帧，或 23.976–144 fps 精确目标帧率。指定目标帧率时节点先构建不低于目标的 2x/4x/8x 稠密网格再最近邻取样（不重复帧）；目标帧率必须高于源帧率且不超过 6 倍 |
| `motion` | 下拉 | `auto` = 有 NVOFA 硬件光流就用硬件，否则 OpenCV DIS；`nvof` / `dis` 强制指定。镜头切换处不产出生成帧 |
| `scene_change_threshold` | FLOAT，0.01–1.0，默认 0.24 | 亮度均值变化超过该阈值即重置时序历史（镜头切换检测）。调高切换更少 |
| `keep_audio` | 下拉 on/off，默认 on | VIDEO 输出保留原音轨：优先从源文件 stream copy,没有源文件时从 AUDIO 输入重编码 |
| `audio` | AUDIO | 可选音轨输入：VIDEO 输入时（keep_audio=on）覆盖原音轨；IMAGE 输入时是唯一的配音方式。无法 stream copy 时重编码为 AAC |
| `runtime_dir` | 字符串 | 覆盖 `dlssg-worker.exe + nvngx.dll + _nvngx.dll + nvngx_dlssg.dll` 所在目录；查找顺序 `DLSS5_FG_RUNTIME_DIR` → `<models>/dlss5/dlssg` → 插件自带 `runtime/dlssg` |
| `wine_prefix` | 字符串 | Linux：覆盖 worker 的 WINEPREFIX（需要 DXVK + vkd3d-proton + DXVK-NVAPI，与增强节点同一套栈） |

| 输出 | 类型 | 含义 |
| --- | --- | --- |
| `images` | IMAGE | 交错后的帧（仅 IMAGE 输入；VIDEO 输入为 None） |
| `video` | VIDEO | 插帧后的视频，交 SaveVideo（VIDEO 输入为流式编码的临时文件对象） |
| `status` | STRING | 单行摘要：引导光流引擎、倍率、帧数、帧率、runtime 路径 |

## 🚀 使用方法

神经渲染参数怎么调（`style` 是独立于上述参数的整体风格档，default/natural/cinematic）：

- **视频降噪 / 去压缩伪影**：`intensity` 保持 1.0；嫌细节被抹升 `structure`，嫌噪点多降 `structure`；
- **人像素材**：`auto_mask=on` + `skin` 默认 2.0，皮肤平滑但保留结构；非人像画面不受影响；
- **结果感觉"过处理"**：先降 `structure`，不够再降 `intensity`；`tone` 一般不用动。

### 示例工作流

- [`examples/ComfyUI-RH-DLSS5_image_api.json`](examples/ComfyUI-RH-DLSS5_image_api.json) — LoadImage → RH_DLSS5Enhance → SaveImage。
- [`examples/ComfyUI-RH-DLSS5_video_api.json`](examples/ComfyUI-RH-DLSS5_video_api.json) — LoadVideo → RH_DLSS5Enhance → SaveVideo（LoadVideo/SaveVideo 需较新 ComfyUI）。
- [`examples/ComfyUI-RH-DLSS5_frame_interp_api.json`](examples/ComfyUI-RH-DLSS5_frame_interp_api.json) — LoadVideo → RH_DLSS5FrameInterpolation（2x）→ SaveVideo。

将 JSON 导入 ComfyUI，提交前把 Load 节点里的占位媒体名替换为真实文件。

帧插值使用建议：24→48 帧率用 `2x` 最稳（每个区间恰好一帧生成帧）；`3x`/`4x` 由级联 2x 构成，耗时与内存成比例上涨。长视频走 VIDEO 输入是边插帧边编码，内存随解码器而非片长增长。

## 队列与缓存

节点每次执行无状态、可安全排队：Linux 每次执行拉起一个 Wine worker（内部锁串行化），Windows 每个 GPU 保持一个桥会话并在尺寸/参数一致时复用。`IS_CHANGED` 对参数 + 输入张量探针 / 源视频元数据做哈希，重复提交相同任务直接命中 ComfyUI 缓存。

## 验证情况

Linux 路线已端到端验证：320x240、48 帧测试片，1x 与 1.5x（输出 480x360），环境为 RTX 4090 / 驱动 580.95.05 + Wine 11.0 + DXVK 3.1 + vkd3d-proton 3.0.1 + DXVK-NVAPI，运行时 DLSSNR 310.8。2026-09 在一台全新 Ubuntu 22.04 容器上按本文档从零安装（含 2b fallback 与分层自检）实测通过。Windows 进程内桥与 MIT 参考实现使用同一协议。

帧插值在同一台机器验证：2x/3x/4x 帧数与源帧落点均以合成运动内容逐帧核对（NVOFA 硬件光流与 DIS 两种引导、音轨 stream copy），400 帧 1080p 2x 实测宿主内存峰值约 20 GB 完成；同一片段在流式改造前的实现会被内存 cgroup 杀掉。

## 🔗 相关链接

[![RunningHub China](https://img.shields.io/badge/RunningHub-China-2F80ED)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International-7B61FF)](https://www.runninghub.ai/?inviteCode=rh-v1367)

- 上游参考（本插件的 Linux/Wine CLI 基础）：<https://github.com/kos94ok/ComfyUI-DLSS5-NR-Linux>
- 上游参考（Windows 进程内桥）：<https://github.com/lisitskyaa/ComfyUI-DLSS5-NR>
- 视频 worker 参考（Merserk 协议）：<https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer>
- 帧生成 worker 与调度参考：<https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation>
- NVIDIA 官方 DLSS SDK（`nvngx_dlss.dll` 来源）：<https://github.com/NVIDIA/DLSS/releases>
- NVIDIA DLSS 仓库：<https://github.com/NVIDIA/DLSS>
- DXVK：<https://github.com/doitsujin/dxvk> · vkd3d-proton：<https://github.com/HansKristian-Work/vkd3d-proton> · DXVK-NVAPI：<https://github.com/jp7677/dxvk-nvapi>

## 📄 协议

插件代码与项目自有原生组件：**MIT**，见 [LICENSE](LICENSE)。`native/src/` 下的桥/host 源码改编自上述 MIT 协议的 ComfyUI-DLSS5-NR 项目。

NVIDIA NGX/DLSS 运行时二进制（`nvngx_dlssnr.dll`、`nvngx_dlss.dll`、`_nvngx.dll`）为 **NVIDIA 专有软件**：不随插件附带、不下载、不再分发，请仅使用你有权使用的构建。DLSS 与 NGX 是 NVIDIA 的商标。

## 🙏 致谢

本项目基于 MIT 协议的参考实现 [ComfyUI-DLSS5-NR-Linux](https://github.com/kos94ok/ComfyUI-DLSS5-NR-Linux)（Linux/Wine CLI，Linux worker 的传输协议改编自其 `dlss5nr_video.py`）与 [ComfyUI-DLSS5-NR](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR)（Windows 进程内桥）。帧生成管线遵循 Merserk 一系的 worker 协议，调度设计参考 [ComfyUI-NVIDIA-DLSS-Frame-Interpolation](https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation)。并依赖 [DXVK](https://github.com/doitsujin/dxvk)、[vkd3d-proton](https://github.com/HansKristian-Work/vkd3d-proton) 与 [DXVK-NVAPI](https://github.com/jp7677/dxvk-nvapi)。感谢 NVIDIA 提供 NGX/DLSS SDK；感谢 [@tetsuoo-online](https://github.com/tetsuoo-online) 报告并定位了 Windows 后端的问题（#1）；感谢 [@MakkiShizu](https://github.com/MakkiShizu) 贡献 Windows 光流与 worker 启动修复（#2）。
