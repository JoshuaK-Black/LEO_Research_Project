# LEO Research Project — Overlay-Only Repo

This repository contains the overlay code and scripts for reproducing the ns-3.45 scenarios:

- `overlay/scratch` – ns-3 C++ scenarios and headers (Q-Routing teacher + RL env, OSPF-lite)
- `overlay/python` – RL agent scripts
- `overlay/scripts` – sweep/aggregation helpers
- `scripts/` – tools to sync and build ns-3
- `docs/` – architecture + reproduction notes
- `third_party/` – placeholder for ns-3 (source clone or submodule)

## Quickstart
```bash
# 1. Clone ns-3.45 into third_party/
git clone https://gitlab.com/nsnam/ns-3-dev.git third_party/ns-3-dev
cd third_party/ns-3-dev && git checkout ns-3.45 && cd ../..

# 2. Sync overlay
./scripts/sync_overlay.sh

# 3. Build
./scripts/build_ns3.sh
```
