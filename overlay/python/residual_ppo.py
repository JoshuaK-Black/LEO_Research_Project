#!/usr/bin/env python3
"""
Residual PPO agent with a 64-128-64 MLP that mixes a teacher action (K index) with a learned policy.
Targets the ns3-gym environment you already run (Discrete(K+1) action space recommended).

Usage (example):
  python3 residual_ppo.py --port 5579 --step-time 0.10 --updates 200 --rollout 256 \
      --alpha 0.05 --beta 0.01 --start-sim 0

Notes:
- Expects observation dict keys: 'mask' (len K), 'nei_feat' (K*F or [K,F]), 'teacher_idx'.
- There may be no explicit 'K' key; K is inferred from len(mask). Teacher action is index K.
- Sends Discrete int in [0..K] where K means "teacher".
"""

import os
import sys
import argparse
from pathlib import Path, PurePath
import subprocess
import time
import threading
import json
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

# ---- metrics helpers ----
class GateMetrics:
    def __init__(self):
        self.margins = []           # p_best - p_teacher (neighbor-only)
        self.tprobs  = []           # p_teacher (neighbor-only)
        self.k_hist  = defaultdict(int)  # {-1: defer, 0..K-1: overrides}
        self.mask_violations = 0
        self.candidate_count = 0
        self.applied_overrides = 0   # reserved for env-accept lines if available
        self.total_steps = 0

    def add_step(self, k, mask_sum, p_best, p_teacher, margin, chosen, mask_ok, teacher_idx):
        self.total_steps += 1
        if mask_ok and chosen != teacher_idx and chosen != -1:
            self.candidate_count += 1
            self.margins.append(margin)
            self.tprobs.append(p_teacher)
        if chosen != -1 and not mask_ok:
            self.mask_violations += 1
        self.k_hist[int(chosen)] += 1
        if (self.total_steps % 10) == 0:
            try:
                print(f"[GATE] step={self.total_steps} K={k} mask_sum={mask_sum} "
                      f"p_best={p_best:.4f} p_teacher={p_teacher:.4f} margin={margin:.4f} "
                      f"chosen={chosen} teacher_idx={teacher_idx} mask_ok={int(mask_ok)}")
            except Exception:
                pass

    def summary(self):
        def pct(x, q):
            if not x:
                return float('nan')
            return float(np.percentile(x, q))
        def r(v):
            return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.4f}"
        cand = max(1, self.candidate_count)
        tot  = max(1, self.total_steps)
        print("==== GATE SUMMARY ====")
        print(f"steps={self.total_steps}  candidates={self.candidate_count}  mask_violations={self.mask_violations}")
        try:
            print(f"coverage_raw={(tot - self.k_hist.get(-1,0)) / tot:.4f}  coverage_candidates={self.candidate_count / tot:.4f}")
        except Exception:
            pass
        print(f"margin_p50={r(pct(self.margins,50))}  p70={r(pct(self.margins,70))}  p80={r(pct(self.margins,80))}  p90={r(pct(self.margins,90))}")
        print(f"teacher_p_p50={r(pct(self.tprobs,50))}  p70={r(pct(self.tprobs,70))}  p80={r(pct(self.tprobs,80))}  p90={r(pct(self.tprobs,90))}")
        print(f"k_hist={dict(sorted(self.k_hist.items()))}")
        print("======================")


def as_scalar(x, default: int = -1) -> int:
    """Return first element as int from scalars/arrays; default on failure."""
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        return int(arr.reshape(-1)[0]) if arr.size else default
    try:
        return int(x)
    except Exception:
        return default


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: [B,K], mask: [B,K] {0,1}
    bad = (mask < 0.5)
    return logits.masked_fill(bad, float("-inf"))


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(masked_logits(logits, mask), dim=dim)


def write_manifest(ckpt_path, args):
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "TG-PPO" if (getattr(args, "train", False) and getattr(args, "beta0", 0) > 0)
                 else "PURE" if getattr(args, "train", False) else "EVAL",
        "updates": getattr(args, "updates", None),
        "rollout": getattr(args, "rollout", None),
        "alpha0": getattr(args, "alpha0", None),
        "alpha_max": getattr(args, "alpha_max", None),
        "alpha_warmup_updates": getattr(args, "alpha_warmup_updates", None),
        "beta0": getattr(args, "beta0", None),
        "beta_decay": getattr(args, "beta_decay", None),
        "alpha_final": getattr(args, "alpha_final", None),
        "seed": getattr(args, "seed", None),
        "ns3_flags": os.getenv("NS3_LAST_CMD", ""),
        "git": subprocess.getoutput("git rev-parse --short HEAD 2>/dev/null"),
    }
    jpath = str(PurePath(ckpt_path).with_suffix(".json"))
    with open(jpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] wrote {jpath}")


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_ckpt(ckpt_path: str, state_dict, args):
    ensure_parent(ckpt_path)
    torch.save(state_dict, ckpt_path)
    try:
        write_manifest(ckpt_path, args)
    except Exception as e:
        print(f"[META] manifest write failed: {e}", file=sys.stderr)


def load_ckpt(load_path: str, model, device):
    print(f"[LOAD] {load_path}")
    obj = torch.load(load_path, map_location=device)
    if isinstance(obj, dict) and ('model' in obj or 'state_dict' in obj):
        state = obj.get('model', obj.get('state_dict'))
    else:
        state = obj
    model.load_state_dict(state, strict=False)


def pack_action(a: int, K: int) -> int:
    # Discrete env: return scalar int in [0..K] (K maps to teacher)
    return int(a)


class ResidualMLP(nn.Module):
    def __init__(self, K: int, feat_dim: int, g_dim: int = 0):
        super().__init__()
        in_dim = K * feat_dim + g_dim
        self.K = K
        self.f = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.policy = nn.Linear(64, K + 1)  # last logit = teacher
        self.value = nn.Linear(64, 1)

    def policy_logits_from_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Given a flattened neighbor feature tensor [B, K*F], return policy logits [B, K+1]."""
        h = self.f(x)
        return self.policy(h)

    def forward(self, nei_feat: torch.Tensor, mask: torch.Tensor, g: torch.Tensor | None = None,
                alpha: float = 0.05, teacher_idx: torch.Tensor | None = None):
        """
        nei_feat: [B,K,F]
        mask:     [B,K]
        g:        [B,g_dim] or None
        teacher_idx: [B] long or None
        """
        B, K, Fdim = nei_feat.shape
        x = nei_feat.reshape(B, K * Fdim)
        if g is not None:
            x = torch.cat([x, g], dim=-1)
        h = self.f(x)
        raw_logits = self.policy(h)  # [B, K+1]

        # Mask invalid neighbors (do not mask teacher at index K)
        neigh_logits = raw_logits[:, :self.K]
        neigh_logits = masked_logits(neigh_logits, mask)
        logits = torch.cat([neigh_logits, raw_logits[:, self.K:self.K + 1]], dim=-1)

        pi_agent = F.softmax(logits, dim=-1)

        # Build teacher one-hot over K+1
        if teacher_idx is not None:
            t_idx = teacher_idx.clamp(min=0)
            one_hot_teacher = F.one_hot(t_idx, num_classes=self.K + 1).float()
        else:
            one_hot_teacher = torch.zeros_like(pi_agent)
            one_hot_teacher[:, self.K] = 1.0

        # Residual mix
        pi_mixed = (1.0 - alpha) * one_hot_teacher + alpha * pi_agent
        v = self.value(h).squeeze(-1)
        return logits, pi_mixed, v


class ResidualPolicyNet(nn.Module):
    """Lazy-initialized wrapper that builds ResidualMLP on first forward pass.
    Accepts either:
      - Tensor shaped [B, K, F] or [K, F]
      - Dict with key 'nei_feat' containing [B, K, F] or [K, F]
    Returns policy logits [B, K+1].
    """
    def __init__(self):
        super().__init__()
        self.inner: ResidualMLP | None = None
        self.K = None
        self.F = None

    def _ensure(self, x: torch.Tensor):
        # x expected [B, K, F] or [K, F]
        if x.dim() == 2:
            K, F = int(x.shape[0]), int(x.shape[1])
        else:
            K, F = int(x.shape[1]), int(x.shape[2])
        if self.inner is None:
            print(f"[DBG] Building inner: nei_feat.shape={tuple(x.shape)}")
            self.K, self.F = K, F
            # Ensure parameters are registered on the same device as input
            self.inner = ResidualMLP(K, F).to(x.device)
            try:
                print(f"[DBG] Inner assigned: {sum(p.numel() for p in self.inner.parameters())} params")
            except Exception:
                pass

    def forward(self, x) -> torch.Tensor:
        # Support dict inputs like {'nei_feat': ..., 'mask': ...}
        if isinstance(x, dict):
            nei = x.get('nei_feat', None)
            if nei is None:
                raise ValueError("ResidualPolicyNet expects 'nei_feat' in input dict")
            x = nei
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self._ensure(x)
        B, K, F = x.shape
        xflat = x.reshape(B, K * F)
        return self.inner.policy_logits_from_flat(xflat)

    


# ---------------- Removed signal-based exit guard ----------------
# Previously used a SIGINT/SIGTERM trap to set a global flag.
# This has been removed to allow KeyboardInterrupt to propagate.


def _parse_reason(info):
    """Extract termination reason from info payload or string."""
    try:
        if isinstance(info, dict):
            return str(info.get("reason", "")).strip()
        s = str(info)
        if "SimulationEnd" in s:
            return "SimulationEnd"
        if "GameOver" in s:
            return "GameOver"
    except Exception:
        pass
    return ""


class IdleGuard:
    def __init__(self, timeout_s: float = 15.0):
        self.timeout = float(timeout_s)
        self._last = time.time()

    def bump(self):
        self._last = time.time()

    def expired(self) -> bool:
        return (time.time() - self._last) > self.timeout


def safe_step(env, action, idle_guard: IdleGuard):
    """
    Wrap env.step(action) and convert sim-end races into a clean terminal step.
    idle_guard.bump() records activity to avoid idle hangs.
    """
    try:
        obs, reward, done, info = env.step(action)
        idle_guard.bump()
        if obs is None:
            print("[AGENT] env closed; marking done")
            return None, 0.0, True, {"reason": "EnvClosed"}
        reason = _parse_reason(info)
        if reason in ("SimulationEnd", "GameOver"):
            done = True
        return obs, reward, done, info
    except Exception:
        # ZMQ may drop right at sim end; treat as terminal.
        return None, 0.0, True, {"reason": "SimulationEnd"}


def sim_time_from_info(info) -> float | None:
    """Best-effort extract of simulated time from env info payload.
    Supports dicts or JSON strings with a 'sim_time' or similar key.
    """
    try:
        if isinstance(info, dict):
            for key in ("sim_time", "simTime", "t", "time"):
                if key in info:
                    try:
                        return float(info[key])
                    except Exception:
                        pass
        if isinstance(info, str):
            try:
                obj = json.loads(info)
                if isinstance(obj, dict):
                    for key in ("sim_time", "simTime", "t", "time"):
                        if key in obj:
                            return float(obj[key])
            except Exception:
                pass
            m = re.search(r'"sim_time"\s*:\s*([-+eE0-9\.]+)', info)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    except Exception:
        pass
    return None


def obs_to_tensors(obs: dict):
    """Convert obs dict to typed tensors and infer K,F.
    Returns: nei [1,K,F], msk [1,K], t [1], K, F
    """
    mask_np = np.asarray(obs.get("mask", []), dtype=np.float32).reshape(-1)
    K = int(mask_np.shape[0])
    nei_np = np.asarray(obs.get("nei_feat", []), dtype=np.float32).reshape(K, -1)
    Fdim = int(nei_np.shape[1])
    t_idx = as_scalar(obs.get("teacher_idx", -1))
    # map invalid to teacher action index (K)
    t_idx = t_idx if t_idx >= 0 else K
    nei = torch.from_numpy(nei_np).to(torch.float32).unsqueeze(0)
    msk = torch.from_numpy(mask_np).to(torch.float32).unsqueeze(0)
    t = torch.tensor([t_idx], dtype=torch.long)
    return nei, msk, t, K, Fdim


def choose_action(model: ResidualMLP, obs: dict, alpha: float):
    nei, msk, t, K, Fdim = obs_to_tensors(obs)
    with torch.no_grad():
        logits, pi, v = model(nei, msk, teacher_idx=t, alpha=alpha)
        dist = torch.distributions.Categorical(probs=pi)
        a = dist.sample()
        logp = dist.log_prob(a)
    return int(a.item()), float(logp.item()), float(v.item()), K, Fdim


class RolloutBuf:
    def __init__(self, T, K, Fdim):
        self.obs_nei = torch.zeros(T, K, Fdim, dtype=torch.float32)
        self.mask = torch.zeros(T, K, dtype=torch.float32)
        self.t_idx = torch.zeros(T, dtype=torch.long)
        self.a = torch.zeros(T, dtype=torch.long)
        self.logp = torch.zeros(T, dtype=torch.float32)
        self.v = torch.zeros(T, dtype=torch.float32)
        self.r = torch.zeros(T, dtype=torch.float32)
        self.done = torch.zeros(T, dtype=torch.float32)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


def ppo_update(model: ResidualMLP, opt: torch.optim.Optimizer, buf: RolloutBuf,
               epochs=4, mb=64, clip=0.2, entw=1e-3, vf_coef=0.5,
               beta=0.01, alpha=0.05, gamma=0.99, lam=0.95, valid_T: int | None = None):
    K = buf.mask.shape[1]
    T = int(valid_T) if valid_T is not None else buf.mask.shape[0]
    # GAE
    adv = torch.zeros(T)
    ret = torch.zeros(T)
    lastgaelam = 0.0
    nextv = 0.0
    nextdone = 0.0
    for t in reversed(range(T)):
        delta = buf.r[t] + gamma * nextv * (1.0 - nextdone) - buf.v[t]
        lastgaelam = delta + gamma * lam * (1.0 - nextdone) * lastgaelam
        adv[t] = lastgaelam
        ret[t] = adv[t] + buf.v[t]
        nextv, nextdone = buf.v[t], buf.done[t]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = torch.randperm(T)
    for _ in range(epochs):
        for start in range(0, T, mb):
            mb_idx = idx[start:start + mb]
            nei = buf.obs_nei[mb_idx]
            msk = buf.mask[mb_idx]
            act = buf.a[mb_idx]
            old_logp = buf.logp[mb_idx]
            adv_ = adv[mb_idx]
            ret_ = ret[mb_idx]
            t_idx = buf.t_idx[mb_idx]

            logits, pi, v = model(nei, msk, teacher_idx=t_idx, alpha=alpha)
            dist = torch.distributions.Categorical(probs=pi)
            logp = dist.log_prob(act)
            ratio = (logp - old_logp).exp()
            s1 = ratio * adv_
            s2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(v, ret_)
            entropy = dist.entropy().mean()

            # Teacher KL reg
            with torch.no_grad():
                onehot_t = F.one_hot(t_idx, num_classes=K + 1).float()
            pi_safe = torch.clamp(pi, 1e-8, 1.0)
            kl_to_teacher = (onehot_t * (onehot_t.add(1e-8).log() - pi_safe.log())).sum(dim=-1).mean()

            loss = policy_loss + vf_coef * value_loss - entw * entropy + beta * kl_to_teacher
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


class RewardNorm:
    def __init__(self, eps: float = 1e-8):
        self.m = 0.0
        self.s2 = 1.0
        self.n = 0
        self.eps = eps

    def norm(self, r: float) -> float:
        self.n += 1
        delta = r - self.m
        self.m += delta / self.n
        self.s2 += delta * (r - self.m)
        std = max((self.s2 / max(1, self.n - 1)) ** 0.5, 1e-3)
        return (r - self.m) / (std + self.eps)


def run_training(env, total_updates=200, steps_per_rollout=256, alpha=0.05, beta=0.01, lr=3e-4,
                 model: ResidualMLP | None = None, K: int | None = None, F: int | None = None,
                 sim_time: float = 0.0):
    # 0) First obs
    obs = env.reset()
    t0 = time.time()

    # 1) Infer K,F and build model/opt if needed
    if (K is None) or (F is None) or (model is None):
        _, _, _, K, F = obs_to_tensors(obs)
        if model is None:
            model = ResidualMLP(K, F)
    Fdim = F
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rnorm = RewardNorm()

    terminated = False
    gstep = 0
    for upd in range(total_updates):
        if terminated:
            break
        buf = RolloutBuf(steps_per_rollout, K, Fdim)
        step = 0
        idle = IdleGuard(timeout_s=20.0)
        while step < steps_per_rollout:
            # act
            a, logp, v, K_infer, F_infer = choose_action(model, obs, alpha)
            assert K_infer == K and F_infer == Fdim, "K/F changed during rollout"

            nei_np = np.asarray(obs["nei_feat"], dtype=np.float32).reshape(K, Fdim)
            msk_np = np.asarray(obs["mask"], dtype=np.float32).reshape(K)
            t_idx = as_scalar(obs.get("teacher_idx", -1))
            t_idx = t_idx if t_idx >= 0 else K

            buf.obs_nei[step] = torch.from_numpy(nei_np)
            buf.mask[step] = torch.from_numpy(msk_np)
            buf.t_idx[step] = torch.tensor(t_idx, dtype=torch.long)
            buf.a[step] = torch.tensor(a, dtype=torch.long)
            buf.logp[step] = torch.tensor(logp, dtype=torch.float32)
            buf.v[step] = torch.tensor(v, dtype=torch.float32)

            # step env safely
            obs, r, done, info = safe_step(env, int(a), idle)
            # Prefer simulated time from info over wallclock
            if sim_time > 0:
                st = sim_time_from_info(info)
                if st is not None and st >= sim_time:
                    print(f"[AGENT] simTime {sim_time}s reached (sim={st:.2f}); closing.")
                    try:
                        env.close()
                    except Exception:
                        pass
                    sys.exit(0)
            buf.r[step] = torch.tensor(rnorm.norm(float(r)), dtype=torch.float32)
            buf.done[step] = torch.tensor(float(done), dtype=torch.float32)
            step += 1
            gstep += 1
            if gstep % 10 == 0:
                try:
                    print(f"[AGENT] step={gstep}")
                except Exception:
                    pass
            if done or idle.expired():
                terminated = True
                break

        # PPO update on collected portion (if any)
        if step > 0:
            ppo_update(model, opt, buf, beta=beta, alpha=alpha, valid_T=step)

        # Anneal for long episodes
        alpha = min(0.40, alpha + 0.01)
        beta = max(0.00, beta * 0.95)

        if (upd + 1) % 10 == 0:
            print(f"[UPD {upd+1}] alpha={alpha:.2f} beta={beta:.3f} Rmean={buf.r.mean().item():.4f}")
    if terminated:
        print("[AGENT] Episode terminated or env finished; shutting down.")
        try:
            env.close()
        except Exception:
            pass
        sys.exit(0)
    return model


def main():
    p = argparse.ArgumentParser(description="Residual PPO (teacher-mixed) for ns3-gym")
    p.add_argument("--port", type=int, default=int(os.getenv("OPEN_GYM_PORT", "5579")))
    p.add_argument("--step-time", type=float, default=0.10)
    p.add_argument("--sim-time", type=float, default=0.0,
                   help="Expected ns-3 simTime in seconds; agent exits when exceeded")
    p.add_argument("--start-sim", type=int, default=0)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--alpha", type=float, default=0.05, help="residual mix toward agent")
    p.add_argument("--beta", type=float, default=0.01, help="teacher KL weight")
    p.add_argument("--lr", type=float, default=3e-4)
    # IL collection / training
    p.add_argument("--il-collect", action="store_true",
                   help="Collect teacher dataset (act as teacher, save (nei_feat,mask,teacher_idx))")
    p.add_argument("--il-episodes", type=int, default=3,
                   help="Episodes to collect per run in --il-collect")
    # --- Paths default to results/RL ---
    p.add_argument("--il-out", type=str, default="results/RL/il_init.pt",
                   help="Where to save IL checkpoint")
    p.add_argument("--il-train", type=str, default=None,
                   help="NPZ path for imitation training (offline)")
    # Schedules
    p.add_argument("--alpha0", type=float, default=0.05)
    p.add_argument("--alpha-max", type=float, default=0.30)
    p.add_argument("--alpha-warmup-updates", type=int, default=100)
    p.add_argument("--beta0", type=float, default=0.02)
    p.add_argument("--beta-decay", type=float, default=0.95)
    p.add_argument("--eval", action="store_true", help="Act only; no updates")
    p.add_argument("--train", action="store_true", help="Enable PPO training mode (after IL or for fine-tuning)")

    p.add_argument("--save", type=str, default="results/RL/ppo_qr_180.pt",
                   help="Path to save checkpoint (.pt)")
    p.add_argument("--load", type=str, default=None,
                   help="Path to load checkpoint (.pt)")
    p.add_argument("--alpha-final", type=float, default=None,
                   help="Eval-time residual blend (teacher/policy)")
    # Residual override gating (eval-only)
    p.add_argument("--alpha-eval", type=float, default=0.30,
                   help="Residual strength when overriding (discrete env uses hard switch)")
    p.add_argument("--conf-thresh", type=float, default=0.60,
                   help="Min confidence margin (p_max - p_second) to allow override")
    p.add_argument("--teacher-prob-thresh", type=float, default=0.50,
                   help="If agent assigns >= this prob to teacher action, do not override")
    p.add_argument("--max-override-rate", type=float, default=0.25,
                   help="Safety cap on override fraction per episode (0..1)")
    p.add_argument("--metrics-out", type=str, default="",
                   help="Path to write a one-line JSON summary of gate metrics.")
    # IL-specific knobs (offline imitation learning)
    p.add_argument("--il-batch", type=int, default=1024, help="IL batch size")
    p.add_argument("--il-epochs", type=int, default=6, help="IL epochs")
    p.add_argument("--il-val-split", type=float, default=0.10, help="IL val split fraction [0,1]")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    p.add_argument("--seed", type=int, default=-1, help="Seed RNGs for reproducible runs")
    args = p.parse_args()

    # ==========================================================
    # OFFLINE IMITATION LEARNING MODE
    # ==========================================================
    if args.il_train:
        print(f"[IL] Training imitation model from {args.il_train}")
        device = torch.device("cpu")
        model = ResidualPolicyNet().to(device)

        data = np.load(args.il_train, allow_pickle=True)
        X, M, Y = data["nei_feat"], data["mask"], data["target"]

        # Simple split
        VAL = float(args.il_val_split)
        n = int((1.0 - VAL) * len(Y))
        X_train, X_val = X[:n], X[n:]
        M_train, M_val = M[:n], M[n:]
        Y_train, Y_val = Y[:n], Y[n:]

        # Normalize potential object arrays into numeric float32/int64
        if isinstance(X_train, np.ndarray) and X_train.dtype == object:
            X_train = np.stack(X_train).astype(np.float32)
        if isinstance(X_val, np.ndarray) and X_val.dtype == object:
            X_val = np.stack(X_val).astype(np.float32)
        if isinstance(M_train, np.ndarray) and M_train.dtype == object:
            M_train = np.stack(M_train).astype(np.float32)
        if isinstance(M_val, np.ndarray) and M_val.dtype == object:
            M_val = np.stack(M_val).astype(np.float32)
        Y_train = np.array(Y_train, dtype=np.int64)
        Y_val = np.array(Y_val, dtype=np.int64)

        # Trigger lazy build by running one dummy forward pass
        try:
            x0 = np.array(X[0], dtype=np.float32)
            K0, F0 = int(x0.shape[0]), int(x0.shape[1])
            dummy_obs = {
                "nei_feat": torch.zeros((K0, F0), dtype=torch.float32),
                "mask": torch.ones((K0,), dtype=torch.float32),
            }
            print("[DBG] Before dummy forward:", sum(p.numel() for p in model.parameters()))
            _ = model(dummy_obs)
            print("[DBG] After dummy forward:", sum(p.numel() for p in model.parameters()))
            print("[DBG] After dummy forward param count:", sum(p.numel() for p in model.parameters()))
            print("[DBG] Model children:", list(model.named_children()))
            for name, param in model.named_parameters():
                try:
                    print(f"[DBG] param {name}: shape={tuple(param.shape)} requires_grad={param.requires_grad}")
                except Exception:
                    print(f"[DBG] param {name}: (shape unavailable)")
        except Exception:
            # fallback if dataset format unexpected
            pass

        BATCH = int(args.il_batch)
        EPOCHS = int(args.il_epochs)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        N = len(Y_train)
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            total, correct = 0, 0
            # mini-batch training
            for start in range(0, N, BATCH):
                end = min(N, start + BATCH)
                xb = torch.from_numpy(X_train[start:end]).float()
                yb = torch.from_numpy(Y_train[start:end]).long()
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * (end - start)
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    correct += int((pred == yb).sum().item())
                    total += (end - start)
            avg_loss = total_loss / max(1, total)
            acc = (correct / max(1, total)) if total > 0 else 0.0
            print(f"[IL] Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} acc={acc:.3f}")

        save_ckpt(args.save, model.state_dict(), args)
        print(f"[IL] Saved imitation checkpoint to {args.save}")
        return

    # Seed all RNGs deterministically if requested (fallback to env vars)
    def set_all_seeds(seed: int):
        if seed is None or seed < 0:
            try:
                seed = int(os.environ.get("TORCH_SEED", "0"))
            except Exception:
                return
        try:
            import random
            random.seed(seed)
        except Exception:
            pass
        try:
            import numpy as _np
            _np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch as _torch
            _torch.manual_seed(seed)
            _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    set_all_seeds(args.seed)

    from ns3gym import ns3env
    env = ns3env.Ns3Env(port=args.port, stepTime=args.step_time, startSim=bool(args.start_sim))
    # One-time space introspection
    try:
        print(f"[SPACE] action={env.action_space} obs={env.observation_space}")
    except Exception:
        pass

    # IL collect path
    if args.il_collect:
        def _wait_first_obs(env, retries: int = 100, sleep_s: float = 0.1):
            """Block until first observation dict is available after reset."""
            import time as _time
            obs = env.reset()
            n = 0
            while (obs is None or not isinstance(obs, dict) or "mask" not in obs) and n < retries:
                _time.sleep(sleep_s)
                try:
                    # Some ns3gym variants expose get_state(); fall back to reset otherwise
                    if hasattr(env, 'get_state'):
                        obs, _, _, _ = env.get_state()
                    else:
                        obs = env.reset()
                except Exception:
                    obs = None
                n += 1
            if obs is None or not isinstance(obs, dict) or "mask" not in obs:
                raise RuntimeError("env did not provide initial observation in time")
            return obs

        def collect_teacher_dataset(env, episodes, out_path):
            feats, masks, targets = [], [], []
            idle = IdleGuard(timeout_s=20.0)
            total = 0
            MAX_STEPS = 100000

            def _valid_obs(o):
                return isinstance(o, dict) and ("mask" in o) and ("nei_feat" in o)

            try:
                for ep in range(episodes):
                    # First obs (blocks until ns-3 sends SimInit/first state)
                    obs = env.reset() if hasattr(env, "reset") else env.step(0)[0]
                    t0 = time.time()
                    idle.bump()
                    done = False
                    while not done:
                        # If env ended or was restarted, bail cleanly
                        if not _valid_obs(obs):
                            print("[IL] invalid or missing obs; ending episode")
                            break

                        # --- collect features from current obs ---
                        mask = np.asarray(obs["mask"], dtype=np.float32)
                        K = int(mask.shape[0])
                        nei = np.asarray(obs["nei_feat"], dtype=np.float32).reshape(K, -1)
                        t_idx = int(obs.get("teacher_idx", -1))
                        tgt = t_idx if t_idx >= 0 else K  # K = “teacher” action

                        feats.append(nei); masks.append(mask); targets.append(tgt)
                        total += 1

                        # --- act (teacher proxy) and advance ---
                        obs, _, done, info = safe_step(env, K, idle)

                        # Check simulated time from info (preferred over wallclock)
                        if args.sim_time > 0:
                            st = sim_time_from_info(info)
                            if st is not None and st >= args.sim_time:
                                print(f"[AGENT] simTime {args.sim_time}s reached (sim={st:.2f}); closing.")
                                try:
                                    env.close()
                                except Exception:
                                    pass
                                sys.exit(0)

                        # hard end-of-sim guard: if step returned no obs, stop now
                        if done or (obs is None):
                            print(f"[IL] terminal step: done={done} reason={_parse_reason(info)}")
                            break

                        if idle.expired():
                            print("[IL] idle timeout; ending episode")
                            break

                        # Safety net ceiling
                        if total >= MAX_STEPS:
                            break
                    # KeyboardInterrupt will propagate and be handled at top-level
            finally:
                np.savez_compressed(out_path,
                                    nei_feat=np.array(feats, dtype=object),
                                    mask=np.array(masks, dtype=object),
                                    target=np.array(targets, dtype=np.int64))
                print(f"[IL] saved {len(targets)} frames to {out_path}")
                try:
                    env.close()
                except Exception:
                    pass

        collect_teacher_dataset(env, args.il_episodes, args.il_out)
        sys.exit(0)

    # IL train path
    if args.il_train:
        def mask_logits_with_teacher(logits: torch.Tensor, mask: torch.Tensor, K: int) -> torch.Tensor:
            very_neg = torch.finfo(logits.dtype).min / 4
            m = (mask <= 0.0)
            logits[:, :K] = torch.where(m, very_neg, logits[:, :K])
            return logits

        def collate_variable_K(batch):
            Ks = [b[3] for b in batch]
            F = batch[0][4]
            Kmax = max(Ks)
            B = len(batch)
            nei = torch.zeros(B, Kmax, F, dtype=torch.float32)
            msk = torch.zeros(B, Kmax, dtype=torch.float32)
            tgt = torch.zeros(B, dtype=torch.long)
            for i, (n, m, t, K_, F_) in enumerate([(b[0], b[1], b[2], b[3], b[4]) for b in batch]):
                nei[i, :K_, :] = torch.from_numpy(n)
                msk[i, :K_] = torch.from_numpy(m)
                tgt[i] = t if t != K_ else Kmax  # remap teacher K_ -> Kmax
            return nei, msk, tgt, Kmax, F

        def il_train(npz_path, epochs=6, batch_size=1024, lr=3e-4, save_path='il_init.pt'):
            data = np.load(npz_path, allow_pickle=True)
            feats = data['nei_feat']
            masks = data['mask']
            tgts = data['target']
            # infer F from first
            K0, F = np.array(feats[0]).shape
            dataset = []
            for i in range(len(tgts)):
                n = np.array(feats[i], dtype=np.float32)
                m = np.array(masks[i], dtype=np.float32)
                K = n.shape[0]
                t = int(tgts[i])
                dataset.append((n, m, t, K, F))

            model = None
            opt = None
            N = len(dataset)
            indices = np.arange(N)
            for ep in range(epochs):
                np.random.shuffle(indices)
                start = 0
                total_loss = 0.0
                total, correct = 0, 0
                while start < N:
                    batch_idx = indices[start:start + batch_size]
                    start += batch_size
                    batch = [dataset[i] for i in batch_idx]
                    nei, msk, tgt, Kmax, F = collate_variable_K(batch)
                    if model is None:
                        model = ResidualMLP(Kmax, F)
                        opt = torch.optim.Adam(model.parameters(), lr=lr)
                    B = nei.shape[0]
                    x = nei.reshape(B, Kmax * F)
                    logits = model.policy_logits_from_flat(x)
                    logits = mask_logits_with_teacher(logits, msk, Kmax)
                    loss = F.cross_entropy(logits, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total_loss += float(loss.item()) * B
                    with torch.no_grad():
                        pred = logits.argmax(dim=1)
                        correct += int((pred == tgt).sum().item())
                        total += B
                print(f"[IL] epoch {ep+1}/{epochs} loss={total_loss/total:.4f} acc={correct/total:.3f}")
            save_ckpt(args.il_out, model.state_dict(), args)
            print(f"[IL] saved warm-start weights to {args.il_out}")
            return args.il_out

        il_train(args.il_train, epochs=6, batch_size=1024, lr=args.lr, save_path=(args.save or 'il_init.pt'))
        env.close()
        sys.exit(0)

    # Eval path: build model from first obs, optionally load weights, act only
    if args.eval:
        obs = env.reset()
        try:
            print("[AGENT_CONNECTED] spaces+reset OK", flush=True)
        except Exception:
            pass
        _, _, _, K, F = obs_to_tensors(obs)
        model = ResidualMLP(K, F)
        if args.load:
            device = torch.device("cpu")
            load_ckpt(args.load, model, device)

        def make_override_policy(model, args):
            import os as _os
            model.eval()

            def policy(obs: dict) -> int:
                # initialize per-episode counters on function object
                if not hasattr(policy, 'step_count'):
                    policy.step_count = 0
                    policy.override_count = 0
                    policy.gate_open_count = 0
                    policy.nei_pick_count = 0
                    policy.mask_empty_count = 0

                nei, msk, t, K, Fdim = obs_to_tensors(obs)
                with torch.no_grad():
                    # alpha=1.0 -> pure agent distribution
                    logits, pi_agent, _ = model(nei, msk, teacher_idx=t, alpha=1.0)
                # numpy
                pi = pi_agent.squeeze(0).cpu().numpy()
                A = len(pi)
                k_idx = max(0, A - 1)

                # teacher action is K (index k_idx)
                teacher_action = k_idx

                # mask handling
                mask_np = np.asarray(obs.get("mask", []), dtype=bool)
                if mask_np.shape[0] != k_idx:
                    if mask_np.shape[0] > k_idx:
                        mask_np = mask_np[:k_idx]
                    else:
                        tmp = np.zeros(k_idx, dtype=bool)
                        tmp[:mask_np.shape[0]] = mask_np
                        mask_np = tmp

                # gating stats
                order = np.argsort(-pi)
                p_max = float(pi[order[0]]) if A > 0 else 0.0
                p_second = float(pi[order[1]]) if A > 1 else 0.0
                margin = p_max - p_second
                p_teacher = float(pi[min(max(teacher_action, 0), k_idx)]) if A > 0 else 0.0

                frac = (policy.override_count / policy.step_count) if policy.step_count > 0 else 0.0
                use_override = (
                    (margin >= args.conf_thresh) and
                    (p_teacher <= args.teacher_prob_thresh) and
                    (frac < args.max_override_rate)
                )

                # action selection
                if use_override and k_idx > 0:
                    policy.gate_open_count += 1
                    cand = pi[:k_idx].copy()  # neighbors only
                    if mask_np.shape[0] == k_idx:
                        cand[~mask_np] = -1.0
                    a_nei = int(np.argmax(cand)) if cand.size else teacher_action
                    if cand.size == 0 or cand[a_nei] <= -0.5:
                        policy.mask_empty_count += 1
                        action = teacher_action
                    else:
                        action = a_nei
                else:
                    action = teacher_action

                # counters
                if action != teacher_action:
                    policy.override_count += 1
                    policy.nei_pick_count += 1
                policy.step_count += 1

                # optional debug
                try:
                    if _os.getenv("RESIDUAL_DEBUG", "0") == "1" and (policy.step_count % 5000 == 0 or use_override):
                        print(
                            f"[DBG] step={policy.step_count} use_override={use_override} "
                            f"margin={margin:.3f} p_teacher={p_teacher:.3f} "
                            f"teacher={teacher_action} chosen={action} mask_true={int(mask_np.sum())}"
                        )
                except Exception:
                    pass

                # GateMetrics logging (neighbor-only stats) — non-intrusive
                try:
                    gm = getattr(policy, '_gm', None)
                    if gm is not None:
                        # neighbor-only renormalized distribution
                        cand_stats = pi[:k_idx].copy() if k_idx > 0 else np.zeros(0, dtype=float)
                        if k_idx > 0 and mask_np.shape[0] == k_idx:
                            cand_stats[~mask_np] = 0.0
                        s = cand_stats.sum() if cand_stats.size else 0.0
                        probs_nei = (cand_stats / s) if s > 0 else cand_stats
                        idx_best = int(np.argmax(probs_nei)) if probs_nei.size else -1
                        p_best = float(probs_nei[idx_best]) if idx_best >= 0 else 0.0
                        teacher_idx = -1
                        try:
                            t_idx_full = int(t.item())
                            teacher_idx = t_idx_full if t_idx_full < k_idx else -1
                        except Exception:
                            teacher_idx = -1
                        p_teacher_nei = float(probs_nei[teacher_idx]) if teacher_idx >= 0 and probs_nei.size else 0.0
                        margin_nei = p_best - p_teacher_nei
                        chosen_nei = action if action < k_idx else -1
                        mask_ok = (chosen_nei == -1) or (0 <= chosen_nei < k_idx and (mask_np[chosen_nei] if chosen_nei >= 0 else True))
                        gm.add_step(k_idx, int(mask_np.sum()) if mask_np.size else 0,
                                    p_best, p_teacher_nei, margin_nei,
                                    chosen_nei, mask_ok, teacher_idx)
                except Exception:
                    pass

                return int(action)

            return policy

        def eval_once(env, policy, step_time):
            idle = IdleGuard(timeout_s=20.0)
            obs = env.reset() if hasattr(env, "reset") else env.step(0)[0]
            t0 = time.time()
            idle.bump()
            done = False
            # Before loop: GateMetrics instance
            gm = GateMetrics()
            try:
                policy._gm = gm
            except Exception:
                pass
            while not done:
                action = policy(obs)
                obs, reward, done, info = safe_step(env, action, idle)
                # Check simulated time
                if args.sim_time > 0:
                    st = sim_time_from_info(info)
                    if st is not None and st >= args.sim_time:
                        print(f"[AGENT] simTime {args.sim_time}s reached (sim={st:.2f}); closing.")
                        try:
                            env.close()
                        except Exception:
                            pass
                        sys.exit(0)
                if idle.expired():
                    done = True
            # Episode summary (gate/open/override diagnostics)
            try:
                sc = getattr(policy, 'step_count', 0)
                oc = getattr(policy, 'override_count', 0)
                go = getattr(policy, 'gate_open_count', 0)
                npk = getattr(policy, 'nei_pick_count', 0)
                me = getattr(policy, 'mask_empty_count', 0)
                rate = (oc / max(1, sc)) if sc else 0.0
                print(
                    f"[EVAL-SUMMARY] steps={sc} gate_open={go} overrides={oc} "
                    f"nei_picks={npk} mask_empty={me} override_rate={rate:.4f}"
                )
            except Exception:
                pass
            # ---- gate summary to JSON file ----
            try:
                gm2 = getattr(policy, "_gm", None)
                if gm2 is not None:
                    import json, numpy as _np, time as _time, os as _os
                    def _pct(x, q):
                        if not x:
                            return float('nan')
                        return float(_np.percentile(x, q))
                    summary = {
                        "ts": _time.time(),
                        "steps": gm2.total_steps,
                        "candidates": gm2.candidate_count,
                        "mask_violations": gm2.mask_violations,
                        "coverage_raw": (gm2.total_steps - gm2.k_hist.get(-1,0)) / max(1, gm2.total_steps),
                        "margin_p50": _pct(gm2.margins,50),
                        "margin_p70": _pct(gm2.margins,70),
                        "margin_p80": _pct(gm2.margins,80),
                        "margin_p90": _pct(gm2.margins,90),
                        "teacher_p50": _pct(gm2.tprobs,50),
                        "teacher_p70": _pct(gm2.tprobs,70),
                        "conf_thresh": getattr(args, 'conf_thresh', None),
                        "teacher_prob_thresh": getattr(args, 'teacher_prob_thresh', None),
                        "max_override_rate": getattr(args, 'max_override_rate', None),
                        "seed": getattr(args, 'seed', None),
                    }
                    line = json.dumps(summary, separators=(",",":"))
                    print(f"[METRICS]{line}", flush=True)
                    if args.metrics_out:
                        d = _os.path.dirname(args.metrics_out)
                        if d:
                            _os.makedirs(d, exist_ok=True)
                        with open(args.metrics_out, "w") as f:
                            f.write(line + "\n")
            except Exception as e:
                try:
                    print(f"[METRICS][ERROR]{e}", flush=True)
                except Exception:
                    pass
            try:
                # After loop: GateMetrics summary
                try:
                    gm.summary()
                except Exception:
                    pass
                env.close()
            except Exception:
                pass

        # Build policy with conservative gate (pps-agnostic)
        policy = make_override_policy(model, args)
        eval_once(env, policy, args.step_time)
        return

    try:
        # Train path: optionally load IL init, then train with schedules
        obs = env.reset(); _, _, _, K, F = obs_to_tensors(obs)
        model = ResidualMLP(K, F)
        # Load IL or previous weights
        if args.load:
            device = torch.device("cpu")
            load_ckpt(args.load, model, device)
        # Teacher-guided schedules
        alpha = args.alpha0
        beta = args.beta0
        alpha_step = (args.alpha_max - args.alpha0) / max(1, args.alpha_warmup_updates)
        model = run_training(env, total_updates=args.updates, steps_per_rollout=args.rollout,
                             alpha=alpha, beta=beta, lr=args.lr,
                             model=model, K=K, F=F, sim_time=args.sim_time)
        if args.save:
            save_ckpt(args.save, model.state_dict(), args)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[USER EXIT] KeyboardInterrupt detected — terminating gracefully.\n")
        sys.exit(0)
