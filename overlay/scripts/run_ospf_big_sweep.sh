#!/usr/bin/env bash
set -euo pipefail

# OSPF Big Sweep runner for ns-3.45
# Runs 5 rounds at PPS 40..80, writes logs and summaries under results/OSPF/Big_Sweep

# Ensure we run from the repo root (where ./ns3 lives)
cd "$(dirname "$0")"

# Output directory (override by exporting OUTROOT before running)
OUTROOT="${OUTROOT:-$HOME/ns-allinone-3.45/ns-3.45/results/OSPF/Big_Sweep}"
mkdir -p "$OUTROOT"

# Simulation knobs (override by exporting before running)
SIM="${SIM:-180}"
MEASURE="${MEASURE:-60}"
SEED="${SEED:-4242}"

# helper: PPS -> IPI (seconds)
ipi_for() {
  case "$1" in
    40) echo 0.025 ;;
    45) echo 0.0222222 ;;
    50) echo 0.02 ;;
    55) echo 0.0181818 ;;
    60) echo 0.0166667 ;;
    65) echo 0.0153846 ;;
    70) echo 0.0142857 ;;
    75) echo 0.0133333 ;;
    80) echo 0.0125 ;;
    *)  echo 0.02 ;;
  esac
}

rounds=(1 2 3 4 5)
pps_list=(40 45 50 55 60 65 70 75 80)

for round in "${rounds[@]}"; do
  for PPS in "${pps_list[@]}"; do
    IPI="$(ipi_for "$PPS")"
    SUM="${OUTROOT}/ospf_pps${PPS}_r${round}.txt"
    LOG="${OUTROOT}/ospf_pps${PPS}_r${round}.log"

    echo "[OSPF] round=${round} pps=${PPS} IPI=${IPI}"
    # Use stdbuf for line-buffered tee output; pipefail to propagate failures
    stdbuf -oL -eL ./ns3 run -v scratch/leo_iridium_ospf -- \
      --simTime="${SIM}" \
      --measureStart="${MEASURE}" \
      --planes=6 --perPlane=11 --numGs=12 \
      --rangeKm=6500 --checkPeriod=0.08 --blackoutMs=150 \
      --islRateMbps=30 --islDelayMs=5 --queuePkts=400 \
      --satAltitudeKm=780 --satInclDeg=86.4 --walkerPhase=2 \
      --flows=200 --pktSize=2048 \
      --helloInterval=3 --deadInterval=12 --helloJitter=0.2 --helloMissK=3 \
      --lsaMinInterval=1.2 --minLsArrival=1 --lsaBackoffMax=5 \
      --lsaFanout=2 --lsaRetransmit=0.6 --lsaRefresh=30 --lsaMaxAge=3600 \
      --fullFlood=1 --ackRetrans=1 \
      --ackDelayBase=0.04 --ackDelayJitter=0.02 \
      --spfDelay=0.3 --spfHold=1.2 \
      --spfBackoffInit=0.25 --spfBackoffMax=5 --spfBackoffDecay=0.5 \
      --fibInstallBase=0.02 --fibInstallJitter=0.02 \
      --ctrlRateKbps=128 --ctrlBucket=4096 \
      --logEveryPkts=0 --logEverySimSec=0 \
      --interPacket="${IPI}" \
      --exportSummary="${SUM}" \
      --tag="OSPF-6x11-R6500-GS12-2048B-pps${PPS}" \
      --RngSeed="${SEED}" --RngRun="${round}" \
      2>&1 | tee "${LOG}"
  done
done

echo "[DONE] OSPF Big Sweep results written to: $OUTROOT"

