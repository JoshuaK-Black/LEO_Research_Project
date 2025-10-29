#!/usr/bin/env python3
import os, sys, re, time, itertools, subprocess, shlex, csv
from datetime import datetime

NS3_BIN = "./ns3"
PROGRAM = 'scratch/leo_qrouting_ip'

# --- Fixed network flags (match OSPF conditions) ---
NET_FLAGS = dict(
    planes=6,
    perPlane=11,
    wrap=1,
    spacingKm=500,
    simTime=180,
    measureStart=60,
    numGs=6,
    rangeKm=600,
    checkPeriod=0.1,
    blackoutMs=200,
    islRateMbps=40,
    islDelayMs=5,
    queuePkts=500,
    flows=200,
    pktSize=1500,
    # interPacket will be set per scenario (e.g., 0.02 for 50pps)
    quietApps=1,
)

# --- Quick helpers ---
def flags_to_cmd(flags: dict) -> str:
    parts = []
    for k, v in flags.items():
        if isinstance(v, bool):
            v = 1 if v else 0
        parts.append(f'--{k}={v}')
    return " ".join(parts)

CTRL_RE = re.compile(
    r'\[CTRL\]\[QR\]\s+tx_pkts=(?P<tx_pkts>\d+)'
    r'\s+tx_payload_B=(?P<tx_payload>\d+)'
    r'\s+tx_wire_B=(?P<tx_wire>\d+)'
    r'\s+rx_pkts=(?P<rx_pkts>\d+)'
    r'\s+rx_payload_B=(?P<rx_payload>\d+)'
    r'\s+rx_wire_B=(?P<rx_wire>\d+)'
    r'\s+\|\s+probe_tx_pkts=(?P<probe_tx>\d+)'
    r'\s+pack_tx_pkts=(?P<pack_tx>\d+)'
    r'\s+fb_tx_pkts=(?P<fb_tx>\d+)'
)

DATA_RE = re.compile(
    r'Nodes:\s+(?P<nodes>\d+)\s+ISLs:\s+(?P<isls>\d+).*?'
    r'TxPkts:\s+(?P<data_tx>\d+)\s+RxPkts:\s+(?P<data_rx>\d+)'
    r'\s+Lost:\s+(?P<lost>\d+)\s+PDR:\s+(?P<pdr>[0-9\.]+)'
    r'\s+AvgDelay\(s\):\s+(?P<delay>[0-9\.eE\+\-]+)\s+AvgThroughput\(Mbps/flow\):\s+(?P<tput>[0-9\.eE\+\-]+)',
    re.DOTALL
)

def parse_metrics(stdout: str) -> dict:
    out = {}
    m1 = CTRL_RE.search(stdout)
    if m1:
        out.update({k:int(v) for k,v in m1.groupdict().items()})
    m2 = DATA_RE.search(stdout)
    if m2:
        out.update({
            'nodes': int(m2.group('nodes')),
            'isls': int(m2.group('isls')),
            'data_tx': int(m2.group('data_tx')),
            'data_rx': int(m2.group('data_rx')),
            'lost': int(m2.group('lost')),
            'pdr': float(m2.group('pdr')),
            'delay_s': float(m2.group('delay')),
            'tput_mbps_per_flow': float(m2.group('tput')),
        })
    return out

def run_one(run_flags: dict, timeout_sec=3600) -> (int, str, str):
    cmd = f'{NS3_BIN} run "{PROGRAM} {flags_to_cmd(run_flags)}"'
    print(f"[RUN] {cmd}")
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out,_ = p.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        p.kill()
        return 124, "", "timeout"
    return p.returncode, out, ""

def main():
    # Sweep definitions: interPacket ~ pps, Q-table knobs, probe controls.
    # Adjust GRID as needed.
    pps_to_ipi = {
        50: 0.02,
        55: 0.0181818,
        60: 0.0166667,
    }
    GRID = dict(
        interPacket=[pps_to_ipi[50]],              # keep 50pps for selection pass
        qAlpha=[0.0, 0.1, 0.2],
        qGamma=[0.9],
        qEpsStart=[0.10, 0.05],
        qEpsFinal=[0.02, 0.00],
        qEpsTau=[20, 30],
        qUpdateStride=[5, 10, 20],
        qProbeInterval=[0.3, 0.2],
        qProbeFanout=[1, 2],
        qPenaltyDropMs=[300, 500, 800],
        forceCleanLinks=[1],   # ensure control delivery
        tag=[f"QR-sel-pps50-{int(time.time())}"],
    )

    # Cartesian product
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = list(itertools.product(*values))

    results = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"sweep_qr_50pps_{ts}.csv"

    for combo in combos:
        run = dict(NET_FLAGS)  # copy
        for k,v in zip(keys, combo):
            run[k] = v
        rc, out, err = run_one(run)
        rec = {k: run[k] for k in run}
        rec['rc'] = rc
        rec.update(parse_metrics(out))
        results.append(rec)
        # incremental write
        if results:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
        print(f"[DONE] rc={rc} PDR={rec.get('pdr')} delay={rec.get('delay_s')} tput={rec.get('tput_mbps_per_flow')} ctrlB={rec.get('tx_wire')}")

    print(f"Saved {len(results)} rows to {out_csv}")

if __name__ == "__main__":
    main()
