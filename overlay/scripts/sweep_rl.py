#!/usr/bin/env python3
import os, sys, re, time, itertools, subprocess, threading, queue, csv, socket, json, argparse
from pathlib import Path
from datetime import datetime

NS3_BIN = "./ns3"
PROGRAM = 'scratch/leo_qrouting_ip'
# Resolve agent path in this priority: $AGENT_PATH -> scratch/agent_ddqn_codex.py -> agent_ddqn_codex.py
AGENT = os.environ.get("AGENT_PATH")
if not AGENT:
    if os.path.exists("scratch/agent_ddqn_codex.py"):
        AGENT = "scratch/agent_ddqn_codex.py"
    else:
        AGENT = "agent_ddqn_codex.py"

# --- Fixed network flags (match OSPF) ---
NET_FLAGS = dict(
    planes=6, perPlane=11, wrap=1, spacingKm=500,
    simTime=180, measureStart=60,
    numGs=12, rangeKm=600, checkPeriod=0.1, blackoutMs=200,
    islRateMbps=40, islDelayMs=5,
    queuePkts=500, flows=200, pktSize=1500,
    quietApps=1, forceCleanLinks=1,
)

TRAIN_FLAGS_BASE = dict(NET_FLAGS)
TRAIN_FLAGS_BASE.update(
    qProbeInterval=0.0,
    qProbeFanout=0,
    rlTrainDst=-1,
    rlEnable=1,
    rlDelta=0.25,
    rlK=6,
    qAlpha=0.0,
    qEpsStart=0.0,
    qEpsFinal=0.0,
    qGamma=0.9,
)
EVAL_FLAGS_BASE = dict(NET_FLAGS)
EVAL_FLAGS_BASE.update(
    simTime=180,
    qProbeInterval=0.0,
    qProbeFanout=0,
    rlTrainDst=-1,
    rlEnable=1,
    rlDelta=0.25,
    rlK=6,
    qAlpha=0.0,
    qEpsStart=0.0,
    qEpsFinal=0.0,
    qGamma=0.9,
)

DEFAULT_TRAIN_TIMEOUT = int(os.environ.get("TRAIN_TIMEOUT_SEC", "10800"))
DEFAULT_EVAL_TIMEOUT = int(os.environ.get("EVAL_TIMEOUT_SEC", "5400"))

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

def flags_to_cmd(flags: dict) -> str:
    parts = []
    for k, v in flags.items():
        if isinstance(v, bool):
            v = 1 if v else 0
        parts.append(f'--{k}={v}')
    return " ".join(parts)

def check_port_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False
    except PermissionError:
        print(f"[WARN] port check skipped (permission denied for port {port})", file=sys.stderr)
        return True

def parse_metrics(stdout: str) -> dict:
    out = {}
    m1 = CTRL_RE.search(stdout)
    if m1:
        out.update({k: int(v) for k, v in m1.groupdict().items()})
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

def stream(proc, prefix, sink_q, tee_file=None):
    for line in iter(proc.stdout.readline, ''):
        msg = f"[{prefix}] {line}"
        sys.stdout.write(msg)
        sys.stdout.flush()
        if tee_file:
            try:
                tee_file.write(msg)
                tee_file.flush()
            except ValueError:
                pass  # Log target closed; ignore late writes
        sink_q.put(line)
    proc.stdout.close()

def run_pair(run_flags: dict, timeout_sec=7200, logs_dir="results/RL/logs", agent_env=None) -> tuple[int, str, dict]:
    run_flags = dict(run_flags)
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = run_flags.get("tag", f"rl-run-{ts}")
    base_tag = tag
    suffix = 1
    while os.path.exists(os.path.join(logs_dir, f"{tag}.log")):
        tag = f"{base_tag}-{suffix}"
        suffix += 1
    if run_flags.get("tag") != tag:
        run_flags["tag"] = tag
    log_path = os.path.join(logs_dir, f"{tag}.log")
    print(f"[LOG] tee → {log_path}")
    with open(log_path, "w") as tee:
        agent_path = AGENT
        if not os.path.exists(agent_path):
            print(f"[ERR] agent not found at '{agent_path}'. Set $AGENT_PATH or move the file.", file=sys.stderr)
            meta = dict(
                tag=tag,
                log_path=log_path,
                ns3_cmd=None,
                agent_cmd=None,
                agent_path=agent_path,
                openGymPort=run_flags.get("openGymPort", 5555),
                agent_env=dict(agent_env) if agent_env else {},
                run_flags=dict(run_flags),
            )
            return 2, "", meta
        port = int(run_flags.get("openGymPort", 5555))
        if not check_port_free(port):
            print(f"[WARN] port {port} busy; attempting anyway (maybe prior agent waiting).", file=sys.stderr)

        ns3_cmd = f'{NS3_BIN} run "{PROGRAM} {flags_to_cmd(run_flags)}"'
        print(f"[NS3] {ns3_cmd}")
        tee.write(f"[NS3] {ns3_cmd}\n"); tee.flush()
        ns3 = subprocess.Popen(ns3_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        q = queue.Queue()
        t1 = threading.Thread(target=stream, args=(ns3, "NS3", q, tee), daemon=True)
        t1.start()

        collected = []
        boot_lines = []
        gym_started = False
        start_deadline = time.time() + 60
        while time.time() < start_deadline:
            if ns3.poll() is not None:
                break
            try:
                line = q.get(timeout=1.0)
                boot_lines.append(line)
                if "[RL] gym_start" in line:
                    gym_started = True
                    break
            except queue.Empty:
                pass

        collected.extend(boot_lines)

        if not gym_started:
            msg = "[HARNESS] gym_start not seen; aborting run before agent launch\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            try:
                tee.write(msg)
                tee.flush()
            except ValueError:
                pass
            try:
                ns3.kill()
            except Exception:
                pass
            t1.join(timeout=5)
            while True:
                try:
                    collected.append(q.get_nowait())
                except queue.Empty:
                    break
            try:
                ns3.wait(timeout=20)
            except Exception:
                pass
            time.sleep(5)
            meta = dict(
                tag=tag,
                log_path=log_path,
                ns3_cmd=ns3_cmd,
                agent_cmd=None,
                agent_path=agent_path,
                openGymPort=port,
                agent_env=dict(agent_env) if agent_env else {},
                run_flags=dict(run_flags),
            )
            return 3, "".join(collected), meta

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["OPEN_GYM_PORT"] = str(port)
        env["PORT"] = str(port)
        if agent_env:
            for key, value in agent_env.items():
                env[key] = str(value)

        hp_keys = [
            "PORT",
            "OPEN_GYM_PORT",
            "LR",
            "EPS_DECAY",
            "EPS_START",
            "EPS_MIN",
            "BATCH",
            "WARMUP",
            "TAU",
            "HIDDEN",
        ]

        hp_parts = [f"{key}={env[key]}" for key in hp_keys if key in env]
        hp_line = "[AGENT_ENV] " + " ".join(hp_parts)
        print(hp_line)
        try:
            tee.write(hp_line + "\n"); tee.flush()
        except ValueError:
            pass

        agent_cmd = f'python3 {shlex_quote(agent_path)}'
        print(f"[AGENT] {agent_cmd}")
        try:
            tee.write(f"[AGENT] {agent_cmd}\n"); tee.flush()
        except ValueError:
            pass
        agent = subprocess.Popen(agent_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

        t2 = threading.Thread(target=stream, args=(agent, "AGENT", q, tee), daemon=True)
        t2.start()

        last_tick = time.time()
        begin = time.time()
        ns_exit_time = None
        bind_failure = False
        try:
            while True:
                try:
                    line = q.get(timeout=1.0)
                    collected.append(line)
                    last_tick = time.time()
                    if ("Cannot bind to tcp://*:" in line) or ("Address already in use" in line):
                        msg = "[HARNESS] agent bind failed — killing run and moving on\n"
                        sys.stdout.write(msg); sys.stdout.flush()
                        try:
                            tee.write(msg); tee.flush()
                        except ValueError:
                            pass
                        bind_failure = True
                        try:
                            agent.kill()
                        except Exception:
                            pass
                        try:
                            ns3.kill()
                        except Exception:
                            pass
                        break
                except queue.Empty:
                    if time.time() - last_tick > 30:
                        hb = "[HEARTBEAT] waiting for output...\n"
                        sys.stdout.write(hb); sys.stdout.flush()
                        last_tick = time.time()
                if ns3.poll() is not None:
                    if ns_exit_time is None:
                        ns_exit_time = time.time()
                    if agent.poll() is not None:
                        break
                    if time.time() - ns_exit_time > 15:
                        msg = "[INFO] ns-3 exited; stopping agent\n"
                        sys.stdout.write(msg); sys.stdout.flush()
                        try:
                            tee.write(msg); tee.flush()
                        except ValueError:
                            pass
                        try:
                            agent.terminate()
                        except Exception:
                            pass
                if bind_failure:
                    break
                if time.time() - begin > timeout_sec:
                    msg = "[TIMEOUT] killing processes\n"
                    sys.stdout.write(msg); sys.stdout.flush()
                    try:
                        tee.write(msg); tee.flush()
                    except ValueError:
                        pass
                    ns3.kill(); agent.kill()
                    break
        finally:
            try:
                agent.wait(timeout=20)
            except Exception:
                try:
                    agent.kill()
                except Exception:
                    pass
            try:
                ns3.wait(timeout=20)
            except Exception:
                try:
                    ns3.kill()
                except Exception:
                    pass
            t1.join(timeout=5)
            t2.join(timeout=5)
            while True:
                try:
                    collected.append(q.get_nowait())
                except queue.Empty:
                    break

            time.sleep(5)

        out = "".join(collected)
        meta = dict(
            tag=tag,
            log_path=log_path,
            ns3_cmd=ns3_cmd,
            agent_cmd=agent_cmd,
            agent_path=agent_path,
            openGymPort=port,
            agent_env=dict(agent_env) if agent_env else {},
            run_flags=dict(run_flags),
        )
        return (ns3.returncode or 0), out, meta

def shlex_quote(s: str) -> str:
    if re.fullmatch(r"[\w\-/\.:]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"

EXTRA_AGENT_KEYS = [
    "HIDDEN",
    "TAU",
    "GAMMA",
    "SEED",
    "REPLAY_CAP",
    "CKPT_EVERY",
]


def prioritize_configs_by_env(configs):
    target_lr = os.environ.get("LR")
    target_eps_decay = os.environ.get("EPS_DECAY")
    target_batch = os.environ.get("BATCH")

    if target_lr is None and target_eps_decay is None and target_batch is None:
        return configs

    def _matches(cfg: dict) -> bool:
        env = cfg.get("train_env", {})
        try:
            if target_lr is not None and float(env.get("LR")) != float(target_lr):
                return False
            if target_eps_decay is not None and float(env.get("EPS_DECAY")) != float(target_eps_decay):
                return False
            if target_batch is not None and int(env.get("BATCH")) != int(target_batch):
                return False
        except (TypeError, ValueError):
            return False
        return True

    matches = [cfg for cfg in configs if _matches(cfg)]
    if not matches:
        return configs
    others = [cfg for cfg in configs if not _matches(cfg)]
    return matches + others


def slugify_value(value) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace('.', 'p').replace('-', 'm')


def gather_agent_env_extras() -> dict:
    extras = {}
    for key in EXTRA_AGENT_KEYS:
        val = os.environ.get(key)
        if val is not None:
            extras[key] = val
    return extras


def build_base_agent_env(extras: dict) -> dict:
    base_lr = float(os.environ.get("BASE_LR", "1e-3"))
    base_eps_decay = float(os.environ.get("BASE_EPS_DECAY", "0.9995"))
    base_batch = int(os.environ.get("BASE_BATCH", "256"))
    train_env = dict(extras)
    train_env.update({
        "MODE": "train",
        "LR": base_lr,
        "EPS_DECAY": base_eps_decay,
        "BATCH": base_batch,
    })
    base_warmup_env = os.environ.get("BASE_WARMUP")
    if base_warmup_env is not None:
        train_env["WARMUP"] = int(base_warmup_env)
    elif "WARMUP" not in train_env:
        train_env["WARMUP"] = 4000 if base_batch >= 512 else 2500

    eval_env = dict(extras)
    eval_env.update({
        "MODE": "eval",
        "LR": base_lr,
        "BATCH": base_batch,
        "EPS_DECAY": 1.0,
        "EPS_START": 0.0,
        "EPS_MIN": 0.0,
    })
    return {"train": train_env, "eval": eval_env}


def build_phase_configs(phase: str) -> list:
    extras = gather_agent_env_extras()
    configs = []
    if phase == "PhaseA":
        lr_values = [5e-4, 1e-3, 2e-3]
        eps_decays = [0.9990, 0.9995, 0.9998]
        batch_sizes = [128, 256, 512]
        idx = 1
        for lr in lr_values:
            for decay in eps_decays:
                for batch in batch_sizes:
                    name = f"A{idx:02d}_lr{slugify_value(lr)}_eps{slugify_value(decay)}_b{batch}"
                    train_env = dict(extras)
                    train_env.update({
                        "MODE": "train",
                        "LR": lr,
                        "EPS_DECAY": decay,
                        "BATCH": batch,
                    })
                    if "WARMUP" not in train_env:
                        train_env["WARMUP"] = 4000 if batch >= 512 else 2500

                    eval_env = dict(extras)
                    eval_env.update({
                        "MODE": "eval",
                        "LR": lr,
                        "BATCH": batch,
                        "EPS_DECAY": 1.0,
                        "EPS_START": 0.0,
                        "EPS_MIN": 0.0,
                    })
                    tag_base = f"{phase}-A{idx:02d}-lr{slugify_value(lr)}-eps{slugify_value(decay)}-b{batch}"
                    configs.append(
                        dict(
                            name=name,
                            flag_overrides={},
                            train_env=train_env,
                            eval_env=eval_env,
                            tag_base=tag_base,
                        )
                    )
                    idx += 1
    elif phase == "PhaseB":
        base_env = build_base_agent_env(extras)
        rl_deltas = [0.20, 0.25]
        rl_ks = [4, 6]
        idx = 1
        for delta in rl_deltas:
            for k in rl_ks:
                name = f"B{idx:02d}_delta{slugify_value(delta)}_k{k}"
                overrides = {"rlDelta": delta, "rlK": k}
                tag_base = f"{phase}-B{idx:02d}-delta{slugify_value(delta)}-k{k}"
                configs.append(
                    dict(
                        name=name,
                        flag_overrides=overrides,
                        train_env=dict(base_env["train"]),
                        eval_env=dict(base_env["eval"]),
                        tag_base=tag_base,
                    )
                )
                idx += 1
    elif phase == "PhaseC":
        base_env = build_base_agent_env(extras)
        base_overrides = {}
        base_delta = os.environ.get("BASE_RLDELTA")
        if base_delta is not None:
            base_overrides["rlDelta"] = float(base_delta)
        base_k = os.environ.get("BASE_RLK")
        if base_k is not None:
            base_overrides["rlK"] = int(base_k)
        drop_vals = [0.3, 0.5]
        hop_vals = [0.0, 0.02]
        idx = 1
        for drop in drop_vals:
            for hop in hop_vals:
                overrides = dict(base_overrides)
                overrides.update({
                    "rlDropPenalty": drop,
                    "rlHopPenalty": hop,
                })
                name = f"C{idx:02d}_drop{slugify_value(drop)}_hop{slugify_value(hop)}"
                tag_base = f"{phase}-C{idx:02d}-drop{slugify_value(drop)}-hop{slugify_value(hop)}"
                configs.append(
                    dict(
                        name=name,
                        flag_overrides=overrides,
                        train_env=dict(base_env["train"]),
                        eval_env=dict(base_env["eval"]),
                        tag_base=tag_base,
                    )
                )
                idx += 1
    else:
        raise ValueError(f"Unknown phase '{phase}'")
    return configs


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    pattern = re.compile(r"ckpt_step(\d+)\.pt$")
    best_path = None
    best_step = -1
    for path in ckpt_dir.glob("ckpt_step*.pt"):
        match = pattern.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def record_summary(meta: dict, run_flags: dict, metrics: dict, rc: int, summary_dir: Path) -> Path:
    summary = dict(
        tag=meta.get("tag"),
        return_code=rc,
        ns3_cmd=meta.get("ns3_cmd"),
        agent=dict(
            path=meta.get("agent_path"),
            cmd=meta.get("agent_cmd"),
            openGymPort=meta.get("openGymPort"),
            env=meta.get("agent_env"),
        ),
        run_flags=dict(sorted(run_flags.items())),
        metrics=metrics,
        log_path=meta.get("log_path"),
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{meta.get('tag')}.summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary_path


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="RL sweep orchestrator with phases")
    parser.add_argument("--phase", choices=["PhaseA", "PhaseB", "PhaseC"], default="PhaseA")
    parser.add_argument("--only", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit number of configs")
    parser.add_argument("--count", type=int, default=None, help="alias for --limit")
    return parser.parse_args()


def stringify_env(env: dict) -> dict:
    return {k: str(v) for k, v in env.items()}


def main():
    args = parse_args()
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    phase = args.phase

    configs = build_phase_configs(phase)
    configs = prioritize_configs_by_env(configs)
    limit = args.limit if args.limit is not None else args.count
    if limit is not None:
        configs = configs[: limit]

    if not configs:
        print(f"[WARN] No configurations to run for {phase}")
        return

    phase_root = Path("results") / "RL" / phase
    train_root = phase_root / "train"
    eval_root = phase_root / "eval"
    train_logs_dir = train_root / "logs"
    eval_logs_dir = eval_root / "logs"
    train_summary_dir = train_root / "summaries"
    eval_summary_dir = eval_root / "summaries"
    ckpt_root = train_root / "checkpoints"

    train_results = []
    eval_results = []
    base_port = int(os.environ.get("BASE_OPEN_GYM_PORT", "5555"))
    port_counter = 0

    for idx, cfg in enumerate(configs, start=1):
        name = cfg["name"]
        overrides = cfg.get("flag_overrides", {})
        tag_base = cfg.get("tag_base") or f"{phase}-{name.replace('_', '-')}"

        print(f"\n[PHASE {phase}] Config {idx}/{len(configs)} → {name}")
        if overrides:
            print("  Run flag overrides:", ", ".join(f"{k}={overrides[k]}" for k in sorted(overrides)))
        else:
            print("  Run flag overrides: <none>")

        ckpt_dir = None
        ckpt_path = None
        meta_train = None

        if args.only in ("both", "train"):
            train_flags = dict(TRAIN_FLAGS_BASE)
            train_flags.update(overrides)
            port = base_port + port_counter
            while not check_port_free(port):
                port_counter += 1
                port = base_port + port_counter
            port_counter += 1
            train_flags["openGymPort"] = port
            train_flags["tag"] = f"{tag_base}-train-{ts}"

            ckpt_dir = ckpt_root / train_flags["tag"]
            agent_env_train = stringify_env({**cfg["train_env"], "CKPT_DIR": ckpt_dir})

            rc_train, out_train, meta_train = run_pair(train_flags, timeout_sec=DEFAULT_TRAIN_TIMEOUT, logs_dir=str(train_logs_dir), agent_env=agent_env_train)
            metrics_train = parse_metrics(out_train)
            summary_train = record_summary(meta_train, meta_train.get("run_flags", train_flags), metrics_train, rc_train, train_summary_dir)

            train_row = dict(meta_train.get("run_flags", train_flags))
            train_row["tag"] = meta_train.get("tag")
            train_row["tag_base"] = tag_base
            train_row["rc"] = rc_train
            for key, value in agent_env_train.items():
                if key not in {"CKPT_DIR"}:
                    train_row[f"agent_{key}"] = value
            train_row.update(metrics_train)
            train_row["summary"] = str(summary_train)
            train_row["phase"] = phase
            train_row["stage"] = "train"
            train_results.append(train_row)

            ckpt_path = find_latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                print(f"  [WARN] No checkpoint found in {ckpt_dir}; skipping evaluation")

        if args.only in ("both", "eval"):
            eval_flags = dict(EVAL_FLAGS_BASE)
            eval_flags.update(overrides)
            port = base_port + port_counter
            while not check_port_free(port):
                port_counter += 1
                port = base_port + port_counter
            port_counter += 1
            eval_flags["openGymPort"] = port
            eval_flags["tag"] = f"{tag_base}-eval-{ts}"

            agent_env_eval = dict(cfg["eval_env"])
            if "CKPT_LOAD" not in agent_env_eval:
                if ckpt_path is None:
                    print("  [WARN] Evaluation skipped (no checkpoint provided)")
                    continue
                agent_env_eval["CKPT_LOAD"] = ckpt_path
            agent_env_eval = stringify_env(agent_env_eval)

            rc_eval, out_eval, meta_eval = run_pair(eval_flags, timeout_sec=DEFAULT_EVAL_TIMEOUT, logs_dir=str(eval_logs_dir), agent_env=agent_env_eval)
            metrics_eval = parse_metrics(out_eval)
            summary_eval = record_summary(meta_eval, meta_eval.get("run_flags", eval_flags), metrics_eval, rc_eval, eval_summary_dir)

            eval_row = dict(meta_eval.get("run_flags", eval_flags))
            eval_row["tag"] = meta_eval.get("tag")
            eval_row["tag_base"] = tag_base
            eval_row["rc"] = rc_eval
            for key, value in agent_env_eval.items():
                if key not in {"CKPT_LOAD"}:
                    eval_row[f"agent_{key}"] = value
            eval_row.update(metrics_eval)
            eval_row["summary"] = str(summary_eval)
            eval_row["phase"] = phase
            eval_row["stage"] = "eval"
            if meta_train is not None:
                eval_row["train_tag"] = meta_train.get("tag")
            eval_results.append(eval_row)

    train_csv = train_root / f"{phase}_train_{ts}.csv"
    eval_csv = eval_root / f"{phase}_eval_{ts}.csv"
    write_csv(train_csv, train_results)
    write_csv(eval_csv, eval_results)

    if train_results:
        print(f"\n[PHASE {phase}] Wrote training records to {train_csv}")
    if eval_results:
        print(f"[PHASE {phase}] Wrote evaluation records to {eval_csv}")


if __name__ == "__main__":
    main()
