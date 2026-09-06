// SPDX-License-Identifier: MIT
// Copyright (c) 2026 ComfyUI-DLSS5-NR contributors
//
// Small Windows transport used by tools/dlss5nr_video.py.  Linux owns video
// decode/encode; this process only loads the project bridge under Wine and
// exchanges RGB float32 frames and FP16 motion vectors over stdin/stdout.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kDnr2 = 0x32524E44;  // DNR2
constexpr std::uint32_t kFrm2 = 0x324D5246;  // FRM2
constexpr std::uint32_t kOut1 = 0x3154554F;  // OUT1
constexpr std::uint32_t kEnd1 = 0x31444E45;  // END1

using InitFn = int(__cdecl*)(int, const wchar_t*, char*, int);
using ProcessFn = int(__cdecl*)(const float*, const std::uint16_t*, float*, int, int, int, int,
                                 int, int, int, float, float, float, float, float,
                                 int, int, char*, int);
using ShutdownFn = void(__cdecl*)();

struct Header {
    std::uint32_t magic, input_width, input_height, output_width, output_height;
    std::uint32_t warmup_frames, frame_count, perf_quality;
    std::uint32_t profile, preset, style, automask, ui_correction;
    float intensity, tone, structure, skin, global_tone;
};
struct FrameHeader {
    std::uint32_t magic, index, reset;
};
struct ReplyHeader {
    std::uint32_t magic, index, ok, float_count;
};

bool ReadExact(void* destination, std::size_t bytes) {
    auto* dst = static_cast<std::uint8_t*>(destination);
    while (bytes != 0) {
        const std::size_t n = std::fread(dst, 1, bytes, stdin);
        if (n == 0) return false;
        dst += n;
        bytes -= n;
    }
    return true;
}

bool WriteExact(const void* source, std::size_t bytes) {
    return std::fwrite(source, 1, bytes, stdout) == bytes && std::fflush(stdout) == 0;
}

std::wstring AbsolutePath(const wchar_t* path) {
    wchar_t buffer[32768] = {};
    const DWORD n = GetFullPathNameW(path, static_cast<DWORD>(std::size(buffer)), buffer, nullptr);
    return n != 0 && n < std::size(buffer) ? std::wstring(buffer, n) : std::wstring(path);
}

HMODULE LoadBridge(const std::wstring& runtime) {
    // The Windows builder keeps project-owned binaries together in
    // native/bin, while the runtime directory contains only the external
    // NGX files and caller shim.  Prefer the bridge next to this executable;
    // retaining the runtime lookup keeps older release ZIPs compatible.
    wchar_t self_buffer[32768] = {};
    const DWORD self_len = GetModuleFileNameW(nullptr, self_buffer,
                                               static_cast<DWORD>(std::size(self_buffer)));
    if (self_len != 0 && self_len < std::size(self_buffer)) {
        std::wstring self(self_buffer, self_len);
        const std::size_t slash = self.find_last_of(L"\\/");
        if (slash != std::wstring::npos) {
            const std::wstring adjacent = self.substr(0, slash + 1) + L"dlss5nr_bridge.dll";
            if (HMODULE module = LoadLibraryW(adjacent.c_str())) return module;
        }
    }
    const std::wstring runtime_path = runtime + L"\\dlss5nr_bridge.dll";
    return LoadLibraryW(runtime_path.c_str());
}

void WriteError(std::uint32_t index, const char* message) {
    ReplyHeader reply{kOut1, index, 0, 0};
    WriteExact(&reply, sizeof(reply));
    const auto length = static_cast<std::uint32_t>(std::min<std::size_t>(std::strlen(message), 65535));
    WriteExact(&length, sizeof(length));
    WriteExact(message, length);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    if (argc != 2) {
        std::fprintf(stderr, "usage: dlss5nr_host.exe RUNTIME_DIRECTORY\n");
        return 2;
    }

    const std::wstring runtime = AbsolutePath(argv[1]);
    SetDllDirectoryW(runtime.c_str());
    HMODULE bridge = LoadBridge(runtime);
    if (!bridge) {
        std::fprintf(stderr, "LoadLibrary(dlss5nr_bridge.dll) failed: %lu (looked next to host and in runtime)\n", GetLastError());
        return 3;
    }
    const auto init = reinterpret_cast<InitFn>(GetProcAddress(bridge, "dlss5nr_init"));
    const auto process = reinterpret_cast<ProcessFn>(GetProcAddress(bridge, "dlss5nr_process_v2"));
    const auto shutdown = reinterpret_cast<ShutdownFn>(GetProcAddress(bridge, "dlss5nr_shutdown"));
    const auto version = reinterpret_cast<const char*(__cdecl*)()>(GetProcAddress(bridge, "dlss5nr_version"));
    const auto gpu = reinterpret_cast<const char*(__cdecl*)()>(GetProcAddress(bridge, "dlss5nr_gpu_name"));
    if (!init || !process || !shutdown) {
        std::fprintf(stderr, "bridge exports are incomplete (need dlss5nr_process_v2)\n");
        FreeLibrary(bridge);
        return 4;
    }

    Header header{};
    if (!ReadExact(&header, sizeof(header)) || header.magic != kDnr2 ||
        header.input_width == 0 || header.input_height == 0 || header.output_width == 0 ||
        header.output_height == 0 || header.input_width > 16384 || header.input_height > 16384 ||
        header.output_width > 16384 || header.output_height > 16384 ||
        header.frame_count == 0 || header.frame_count > 1000000) {
        std::fprintf(stderr, "invalid DNR2 header\n");
        FreeLibrary(bridge);
        return 5;
    }
    if (header.warmup_frames > header.frame_count) {
        std::fprintf(stderr, "invalid DNR2 warmup frame count\n");
        FreeLibrary(bridge);
        return 5;
    }
    const std::uint32_t output_long = std::max(header.output_width, header.output_height);
    const std::uint32_t output_short = std::min(header.output_width, header.output_height);
    if (output_long > 7680 || output_short > 4320) {
        std::fprintf(stderr, "DNR2 output exceeds the DLSSNR 7680x4320 envelope\n");
        FreeLibrary(bridge);
        return 5;
    }
    const std::uint64_t input_pixels = static_cast<std::uint64_t>(header.input_width) * header.input_height;
    const std::uint64_t output_pixels = static_cast<std::uint64_t>(header.output_width) * header.output_height;
    if (input_pixels > (1ull << 28) || output_pixels > (1ull << 28)) {
        std::fprintf(stderr, "frame stream is too large\n");
        FreeLibrary(bridge);
        return 6;
    }

    char error[4096] = {};
    int gpu_index = 0;
    char gpu_env[32] = {};
    const DWORD gpu_env_len = GetEnvironmentVariableA("DLSS5NR_GPU_INDEX", gpu_env, sizeof(gpu_env));
    if (gpu_env_len > 0 && gpu_env_len < sizeof(gpu_env)) gpu_index = std::atoi(gpu_env);
    if (gpu_index < 0) gpu_index = 0;
    if (!init(gpu_index, runtime.c_str(), error, static_cast<int>(sizeof(error)))) {
        std::fprintf(stderr, "DLSS5 init failed: %s\n", error[0] ? error : "unknown error");
        FreeLibrary(bridge);
        return 7;
    }
    const std::size_t input_values = static_cast<std::size_t>(input_pixels) * 3;
    const std::size_t output_values = static_cast<std::size_t>(output_pixels) * 3;
    const std::size_t motion_values = static_cast<std::size_t>(input_pixels) * 2;
    std::vector<float> input(input_values), output(output_values);
    std::vector<std::uint16_t> motion(motion_values);
    std::fprintf(stderr, "DLSS5 host ready: %s on %s (%ux%u -> %ux%u, perf=%d)\n",
                 version ? version() : "unknown", gpu ? gpu() : "unknown",
                 header.input_width, header.input_height, header.output_width, header.output_height,
                 header.perf_quality);

    for (std::uint32_t expected = 0; expected < header.frame_count; ++expected) {
        FrameHeader frame{};
        if (!ReadExact(&frame, sizeof(frame)) || frame.magic != kFrm2 || frame.index != expected ||
            !ReadExact(input.data(), input.size() * sizeof(float)) ||
            !ReadExact(motion.data(), motion.size() * sizeof(std::uint16_t))) {
            std::fprintf(stderr, "invalid frame %u\n", expected);
            shutdown();
            FreeLibrary(bridge);
            return 8;
        }
        std::memset(error, 0, sizeof(error));
        const int ok = process(input.data(), motion.data(), output.data(),
                               static_cast<int>(header.input_width), static_cast<int>(header.input_height),
                               static_cast<int>(header.output_width), static_cast<int>(header.output_height),
                               static_cast<int>(header.style), static_cast<int>(header.preset),
                               static_cast<int>(header.perf_quality),
                               header.intensity, header.tone, header.structure, header.skin,
                               header.global_tone,
                               static_cast<int>(header.automask), frame.reset ? 1 : 0,
                               error, static_cast<int>(sizeof(error)));
        if (!ok) {
            const char* message = error[0] ? error : "DLSS5 frame failed";
            std::fprintf(stderr, "DLSS5 frame %u failed: %s\n", expected, message);
            WriteError(expected, message);
            shutdown();
            FreeLibrary(bridge);
            return 9;
        }
        const ReplyHeader reply{kOut1, expected, 1, static_cast<std::uint32_t>(output_values)};
        if (!WriteExact(&reply, sizeof(reply)) || !WriteExact(output.data(), output.size() * sizeof(float))) {
            shutdown();
            FreeLibrary(bridge);
            return 10;
        }
    }
    const std::uint32_t done = kEnd1;
    WriteExact(&done, sizeof(done));
    shutdown();
    FreeLibrary(bridge);
    return 0;
}
