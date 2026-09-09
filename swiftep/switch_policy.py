"""Moebius TP <-> EP switch policy + workload simulation.

Reproduces the paper's key motivation (Sec 2.1): the optimal parallelism for MoE
decode crosses over as the active batch size changes -- TP wins at low
concurrency, EP at high concurrency -- so a runtime switch beats pinning either
layout on bursty / RL-rollout workloads.

The decode-step latency model follows the structural arguments in Sec 2.1:
  * TP: per-layer All-Reduce ships the full hidden state (grows with B) and the
        MoE GEMMs see the full batch B on every rank.
  * EP: the All-to-All carries only routed tokens but pays a small-message
        dispatch floor that dominates at low B; the MoE GEMMs see B/P per rank.

Run:
    python swiftep/switch_policy.py
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyModel:
    """Per-step decode latency coefficients (arbitrary units).

    Args:
        beta: MoE-compute cost per token.
        alpha_ar: All-Reduce cost per token (TP only).
        alpha_a2a: All-to-All cost per routed token (EP only).
        floor: EP dispatch floor (small-message overhead), independent of B.
        p: number of ranks in the switching group.
    """

    beta: float = 1.0
    alpha_ar: float = 0.6
    alpha_a2a: float = 0.15
    floor: float = 200.0
    p: int = 8

    def tp(self, b: float) -> float:
        return (self.alpha_ar + self.beta) * b

    def ep(self, b: float) -> float:
        return self.floor + self.alpha_a2a * b + self.beta * b / self.p

    def crossover(self) -> float:
        """Batch size where TP and EP latencies are equal."""
        denom = (self.alpha_ar + self.beta) - (self.alpha_a2a + self.beta / self.p)
        return self.floor / denom


@dataclass
class SwitchPolicy:
    """Hysteretic policy: switch only when the other mode is better by a margin.

    Args:
        model: latency model.
        margin: fractional advantage required before switching (hysteresis).
    """

    model: LatencyModel
    margin: float = 0.1

    def target(self, b: float, current: str) -> str:
        l_cur = self.model.tp(b) if current == "tp" else self.model.ep(b)
        l_alt = self.model.ep(b) if current == "tp" else self.model.tp(b)
        if (l_cur - l_alt) / l_cur > self.margin:
            return "ep" if current == "tp" else "tp"
        return current


def simulate(load: np.ndarray, model: LatencyModel, switch_cost: float = 0.0):
    """Return per-step latency for fixed-TP, fixed-EP, and adaptive modes.

    Args:
        load: array of batch sizes over time.
        model: latency model.
        switch_cost: one-time latency cost of a switch (in the same units as the
            per-step latency).
    """
    fixed_tp = model.tp(load)
    fixed_ep = model.ep(load)

    policy = SwitchPolicy(model)
    adaptive = np.empty_like(load)
    cur = "tp" if model.tp(load[0]) <= model.ep(load[0]) else "ep"
    for t, b in enumerate(load):
        nxt = policy.target(b, cur)
        if nxt != cur:
            adaptive[t] = (model.tp(b) if nxt == "tp" else model.ep(b)) + switch_cost
            cur = nxt
        else:
            adaptive[t] = model.tp(b) if cur == "tp" else model.ep(b)
    return fixed_tp, fixed_ep, adaptive


def rollout_load(steps: int, peak: int, tail: int, seed: int = 0) -> np.ndarray:
    """RL-rollout shape: a high-concurrency burst decaying to a long tail."""
    rng = np.random.default_rng(seed)
    # Exponential decay from peak to tail, with per-sample-length noise.
    decay = np.exp(-np.linspace(0, 6.0, steps))
    load = tail + (peak - tail) * decay
    return np.maximum(tail, (load * rng.lognormal(0, 0.15, steps)).astype(float))


def bursty_load(steps: int, low: int, high: int, period: int, seed: int = 0) -> np.ndarray:
    """Bursty online-serving shape: alternating quiet periods and bursts."""
    base = np.zeros(steps)
    for i in range(steps):
        base[i] = high if (i // period) % 2 == 0 else low
    return base


def main() -> None:
    model = LatencyModel(p=8)
    x = model.crossover()
    print(f"LatencyModel(p={model.p}): TP/EP crossover at B={x:.0f}")
    for b in (16, 32, 64, 128, 256, 512, 1024):
        print(f"  B={b:5d}: TP={model.tp(b):8.1f}  EP={model.ep(b):8.1f}  "
              f"better={'TP' if model.tp(b) < model.ep(b) else 'EP'}")

    for name, load in [
        ("RL rollout (burst -> tail)", rollout_load(200, 1024, 8)),
        ("bursty online", bursty_load(200, 8, 1024, 25)),
    ]:
        ftp, fep, adp = simulate(load, model, switch_cost=200.0)
        best_static = min(ftp.sum(), fep.sum())
        print(f"\n{name}: total latency")
        print(f"  fixed TP : {ftp.sum():,.0f}")
        print(f"  fixed EP : {fep.sum():,.0f}")
        print(f"  adaptive : {adp.sum():,.0f}  "
              f"(vs best static {adp.sum()/best_static:.2f}x)")


if __name__ == "__main__":
    main()
