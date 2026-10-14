#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="${1:-Release}"

case "${BUILD_TYPE}" in
    Debug)          BUILD_DIR="build-debug" ;;
    Release)        BUILD_DIR="build" ;;
    RelWithDebInfo) BUILD_DIR="build-reldbg" ;;
    MinSizeRel)     BUILD_DIR="build-minsz" ;;
    *) echo "Usage: $0 [Debug|Release|RelWithDebInfo|MinSizeRel]" >&2; exit 1 ;;
esac
JOBS="${JOBS:-$(nproc)}"

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo ""
echo "Binary: ${BUILD_DIR}/QuickCFD"
