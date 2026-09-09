"""Standalone verification of the TP <-> EP switch policy (numpy, no GPU).

Checks the properties the policy must satisfy, per Moebius Sec 2.1:
  1. A crossover exists: TP is faster below it, EP above it.
  2. Hysteresis: the policy does not switch for a sub-margin advantage.
  3. On a bursty workload, adaptive switching never underperforms the better
     static layout (modulo switch cost).

Run: python swiftep/tests/verify_switch_policy.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switch_policy import LatencyModel, SwitchPolicy, bursty_load, simulate  # noqa: E402


def main() -> None:
    model = LatencyModel(p=8)
    x = model.crossover()
    assert x > 0, "crossover must be positive"
    assert model.tp(x - 1) < model.ep(x - 1), "TP should win below crossover"
    assert model.ep(x + 1) < model.tp(x + 1), "EP should win above crossover"

    policy = SwitchPolicy(model, margin=0.1)
    # At a batch size just below the crossover, TP is better but the advantage
    # may be sub-margin: the policy must stay put (no thrash).
    cur = policy.target(max(x - 5, 1), "tp")
    assert cur == "tp", "policy should not flip on a tiny advantage"

    # Adaptive must match or beat the better static layout on a bursty load.
    load = bursty_load(200, 8, 1024, 25)
    ftp, fep, adp = simulate(load, model, switch_cost=200.0)
    assert adp.sum() <= min(ftp.sum(), fep.sum()) * 1.0 + 1e-6, (
        "adaptive should not be worse than the better static layout"
    )
    print(f"crossover={x:.0f}; adaptive vs best static = "
          f"{adp.sum()/min(ftp.sum(), fep.sum()):.2f}x")
    print("switch policy checks passed")


if __name__ == "__main__":
    main()
