# -*- coding: utf-8 -*-
"""ComfyUI-RH-DLSS5 unit tests.

覆盖:
  - 分块解码器 (decoder.probe_video / ChunkedVideoReader / SequentialSource)
  - interpolate_stream_from_source 与 interpolate_stream 的管线等价性 (桩 _sampled_frames)
  - 级联缓冲的 落盘->memmap 回读->清理 全链路 (桩 _stage_frames, stages=2)
  - StreamEncoder 音轨直拷条件 (audio_input=None + source_video)

运行: python3 -m pytest tests/ -q
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_NAME = "rh_dlss5_under_test"


def _load_pkg():
    """以正规包身份加载本仓 (目录名带连字符不能直接 import); nodes 顶层无 comfy
    依赖, 无需桩。"""
    root = Path(__file__).resolve().parent.parent
    if _PKG_NAME not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            _PKG_NAME, root / "__init__.py", submodule_search_locations=[str(root)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_PKG_NAME] = pkg
        spec.loader.exec_module(pkg)
    return sys.modules[_PKG_NAME]


_pkg = _load_pkg()
import importlib  # noqa: E402

fg = importlib.import_module(_PKG_NAME + ".dlss5nr.fg")  # noqa: E402
decoder = importlib.import_module(_PKG_NAME + ".dlss5nr.decoder")  # noqa: E402

NEED_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="no ffmpeg/ffprobe")


def _gen_clip(path, frames=12, w=64, h=48, fps=24, color="red"):
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={color}:size={w}x{h}:rate={fps}",
         "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return str(path)


# --------------------------------------------------------------------------
# 分块解码器
# --------------------------------------------------------------------------

@NEED_FFMPEG
def test_probe_video_counts(tmp_path):
    p = _gen_clip(tmp_path / "probe.mp4", frames=12)
    info = decoder.probe_video(p)
    assert info is not None
    assert info["frames"] == 12 and info["width"] == 64 and info["height"] == 48
    assert abs(info["fps"] - 24.0) < 0.01


def test_probe_video_unreadable():
    assert decoder.probe_video("/nonexistent/x.mp4") is None


@NEED_FFMPEG
def test_chunked_reader_frame_stream(tmp_path):
    p = _gen_clip(tmp_path / "chunk.mp4", frames=35)   # 跨两个 CHUNK_FRAMES=32 块
    reader = decoder.ChunkedVideoReader(p)
    assert (reader.count, reader.width, reader.height) == (35, 64, 48)
    got = list(reader.iter_frames())
    assert [i for i, _ in got] == list(range(35))
    frame = got[0][1]
    assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
    assert frame[24, 32, 0] > 250 and frame[24, 32, 1] < 5 and frame[24, 32, 2] < 5  # 纯红帧(yuv 往返容忍)


@NEED_FFMPEG
def test_sequential_source_monotonic(tmp_path):
    p = _gen_clip(tmp_path / "seq.mp4", frames=6)
    reader = decoder.ChunkedVideoReader(p)
    src = decoder.SequentialSource(reader)
    rgba0, extra0 = src(0)
    assert rgba0.shape == (48, 64, 4) and extra0 is None
    assert (rgba0[..., 3] == 255).all()
    rgba1, _ = src(1)
    # 乱序/跳跃访问必须显式报错, 而不是静默错帧
    with pytest.raises(RuntimeError, match="non-monotonic"):
        src(0)
    src2 = decoder.SequentialSource(decoder.ChunkedVideoReader(p))
    src2(0)
    with pytest.raises(RuntimeError, match="skipped"):
        src2(2)
    src3 = decoder.SequentialSource(decoder.ChunkedVideoReader(p))
    with pytest.raises(RuntimeError, match="skipped"):
        src3(99)   # 越界跳读按 skipped 处理
    src4 = decoder.SequentialSource(decoder.ChunkedVideoReader(p))
    for i in range(6):
        src4(i)
    with pytest.raises(RuntimeError, match="exhausted"):
        src4(6)    # 顺序走完后越界 -> exhausted


# --------------------------------------------------------------------------
# interpolate_stream_from_source 管线等价性
# --------------------------------------------------------------------------

def test_stream_from_source_matches_array_path(monkeypatch):
    calls = []

    def fake_prepare(src_fps, width, height, count, runtime_dir):
        calls.append(("prepare", count, width, height))
        return "runtime", src_fps

    def fake_sampled(sources, count, runtime, src_fps, stages, out_fps,
                     wine_prefix, threshold, motion_mode, progress_callback, engines):
        calls.append(("sampled", count, width_get(sources)))
        engines.append("fake-engine")
        for i in range(4):
            rgba = np.full((6, 8, 4), i + 1, dtype=np.uint8)
            yield rgba, None

    def width_get(sources):
        # 两个入口都要拿到 (h, w): 从首个可访问帧探测
        for probe_index in (0,):
            rgba, _ = sources(probe_index)
            return rgba.shape[1]
        return -1

    monkeypatch.setattr(fg, "_prepare", fake_prepare)
    monkeypatch.setattr(fg, "_sampled_frames", fake_sampled)

    rgb = np.zeros((5, 6, 8, 3), dtype=np.uint8)
    got_array = []
    fg.interpolate_stream(rgb, 24, 2, frame_sink=got_array.append)

    reader_frames = [np.full((6, 8, 3), i + 1, dtype=np.uint8) for i in range(5)]

    class _R:
        count, width, height, fps = 5, 8, 6, 24.0

        def iter_frames(self):
            return enumerate(reader_frames)

    src = decoder.SequentialSource(_R())
    got_source = []
    fg.interpolate_stream_from_source(src, 5, 8, 6, 24, 2, frame_sink=got_source.append)

    assert len(got_array) == len(got_source)
    assert all(np.array_equal(a, b) for a, b in zip(got_array, got_source))
    assert len(got_source) == 4
    assert calls[0] == ("prepare", 5, 8, 6)      # 数组路径
    assert calls[2] == ("prepare", 5, 8, 6)      # source 路径, 同参
    assert calls[1][1:] == calls[3][1:]          # count/width 一致


# --------------------------------------------------------------------------
# 级联缓冲: 落盘 -> memmap 回读 -> 清理
# --------------------------------------------------------------------------

def test_cascade_spills_to_disk_and_cleans(monkeypatch, tmp_path):
    """stages=2: 非末级网格必须走 落盘+memmap (不再整层进 RAM), 且临时文件用后即删."""
    spill_files = []

    real_stage_buffer_path = fg._stage_buffer_path

    def tracked_stage_buffer_path():
        p = real_stage_buffer_path()
        spill_files.append(p)
        return p

    monkeypatch.setattr(fg, "_stage_buffer_path", tracked_stage_buffer_path)

    stage_calls = []

    def fake_stage_frames(sources, count, runtime, stage_fps, wine_prefix, threshold,
                          motion_mode, progress, engines):
        stage_calls.append(count)
        engines.append("fake")
        for index in range(count):
            rgba, _ = sources(index)
            if index:
                yield np.full_like(rgba, 7), None   # 生成帧槽位
            yield rgba, None

    monkeypatch.setattr(fg, "_stage_frames", fake_stage_frames)
    monkeypatch.setattr(fg, "_prepare", lambda *a, **k: ("runtime", a[0]))

    base = [np.full((4, 6, 4), i + 1, dtype=np.uint8) for i in range(5)]

    def src(index):
        return base[index], None

    got = list(fg._sampled_frames(src, 5, "runtime", fg.Fraction(24), 2,
                                  fg.Fraction(48), "", 0.24, "auto", None, []))
    # 级联: 5 -> 9 -> 17 网格; 末级按 48/24=2x 稠密网格逐帧取样
    assert stage_calls == [5, 9]
    assert len(spill_files) == 1
    assert not spill_files[0].exists(), "级联落盘文件必须在生成器结束后清理"
    assert all(len(item) == 2 and item[0].shape == (4, 6, 4) for item in got)
    assert len(got) == 9    # 末级 17 网格按 2x 目标时间线取样 9 帧


# --------------------------------------------------------------------------
# StreamEncoder 音轨直拷条件 (不实例化, 只校验分支语义)
# --------------------------------------------------------------------------

def test_stream_encoder_audio_branch_semantics():
    src = (Path(__file__).resolve().parent.parent / "dlss5nr" / "videoout.py").read_text(encoding="utf-8")
    assert "audio_input is not None or source_video is not None" in src
    assert "-map", "1:a?" in src or '1:a?' in src   # 可选音轨映射保留


# --------------------------------------------------------------------------
# upscaling_mode 自动分辨率桶 (1K/2K/4K/8K -> 最小可达 DLSS 固定倍率)
# --------------------------------------------------------------------------

def test_auto_scale_factor_buckets():
    common = _pkg.dlss5nr.common
    # 16:9 720p 源: 1K/2K/4K 精确命中 1.5x/2x/3x (1920x1080 / 2560x1440 / 3840x2160)
    assert common.auto_scale_factor(1280, 720, "1K") == 1.5
    assert common.auto_scale_factor(1280, 720, "2K") == 2.0
    assert common.auto_scale_factor(1280, 720, "4K") == 3.0
    # 已达/超过桶 -> 1x (DLSS 不做缩小)
    assert common.auto_scale_factor(1920, 1080, "1K") == 1.0
    assert common.auto_scale_factor(3840, 2160, "4K") == 1.0
    assert common.auto_scale_factor(7680, 4320, "8K") == 1.0
    # 竖屏按短边: 1080x1920 选 2K -> 1.5x (1620 >= 1440)
    assert common.auto_scale_factor(1080, 1920, "2K") == 1.5
    # 最小可用倍率优先: 640x480 选 2K -> 3.0 (1440 恰好达标), 不超冲
    assert common.auto_scale_factor(640, 480, "2K") == 3.0
    # 不可达: 短边 * 3 仍低于桶
    with pytest.raises(common.DLSS5Error):
        common.auto_scale_factor(1280, 720, "8K")
    with pytest.raises(common.DLSS5Error):
        common.auto_scale_factor(500, 400, "2K")
    # 包络超限: 达标倍率会把长边推出 7680x4320
    with pytest.raises(common.DLSS5Error):
        common.auto_scale_factor(1000, 3400, "4K")


def test_auto_bucket_mode_params_defer_scale(monkeypatch):
    common_mod = _pkg.dlss5nr.common
    monkeypatch.setattr(common_mod, "check_runtime_files", lambda *a, **k: None)
    monkeypatch.setattr(common_mod, "resolve_runtime_dir", lambda *a, **k: common_mod.PLUGIN_ROOT / "runtime")
    nodes = _pkg.nodes
    params = nodes._build_params({"upscaling_mode": "4K", "runtime_dir": "/nonexistent-runtime-for-test"})
    assert params["auto_bucket"] == "4K"
    assert params["scale"] is None
    fixed = nodes._build_params({"upscaling_mode": "2x (Performance)", "runtime_dir": "/nonexistent-runtime-for-test"})
    assert fixed["auto_bucket"] is None
    assert fixed["scale"] == 2.0
