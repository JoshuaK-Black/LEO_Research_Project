#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3="$ROOT/third_party/ns-3-dev"

if [[ ! -d "$NS3" ]]; then
  echo "ERROR: ns-3 not found at $NS3."
  echo "Clone ns-3.45 into third_party/ns-3-dev first."
  exit 1
fi

cd "$NS3"
echo "[INFO] Cleaning previous build cache..."
rm -rf cmake-cache
mkdir -p cmake-cache && cd cmake-cache

echo "[INFO] Configuring with CMake..."
cmake .. -G Ninja

echo "[INFO] Building ns-3..."
ninja -j$(nproc)

echo "[DONE] ns-3 build complete."
