#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3="$ROOT/third_party/ns-3-dev"

if [[ ! -d "$NS3" ]]; then
  echo "ERROR: ns-3 not found at $NS3."
  echo "Clone ns-3.45 into third_party/ns-3-dev first."
  exit 1
fi

echo "[INFO] Syncing overlay into ns-3 tree..."
rsync -av --delete "$ROOT/overlay/" "$NS3/"

echo "[DONE] Overlay synced into: $NS3"
