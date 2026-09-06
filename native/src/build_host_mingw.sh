#!/usr/bin/env bash
set -euo pipefail

# Linux-side rebuild helper. The bridge and host both build fine with the
# distro MinGW-w64 cross compiler, but std::mutex requires the posix-thread
# variant (the default win32 model drops it). Does not touch any NVIDIA
# runtime files.
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
compiler=${CXX:-x86_64-w64-mingw32-g++-posix}
if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "error: $compiler not found; install a MinGW-w64 posix toolchain" >&2
    exit 1
fi
mkdir -p "$root_dir/native/bin"
"$compiler" -std=c++17 -O2 -shared -static -static-libgcc -static-libstdc++ \
    "$root_dir/native/src/dlss5nr_bridge.cpp" \
    -o "$root_dir/native/bin/dlss5nr_bridge.dll" \
    -ld3d12 -ldxgi -lole32 -luuid -ldwmapi -luser32 -lkernel32
echo "built $root_dir/native/bin/dlss5nr_bridge.dll"
"$compiler" -std=c++17 -O2 -municode -static -static-libgcc -static-libstdc++ \
    "$root_dir/native/src/dlss5nr_host.cpp" \
    -o "$root_dir/native/bin/dlss5nr_host.exe" \
    -lole32 -luser32 -lkernel32
echo "built $root_dir/native/bin/dlss5nr_host.exe"
