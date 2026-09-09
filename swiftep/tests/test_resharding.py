"""Moebius expert-weight resharding correctness tests (pure logic, CPU).

Validates the EP <-> TP resharding math from Moebius (arXiv 2606.26607) Sec 3.1
without multiple GPUs. The all-to-all reshape/permute logic is simulated
in-process and checked against the expected per-rank layout for BOTH weight
tensors:

  * W13 (gate+up, shard the 2I dim = dim 1 of (E, 2I, H))
  * W2  (down,      shard the  I dim = dim 2 of (E, H, I))

Run: python -m pytest swiftep/tests/test_resharding.py -v
"""

import torch


def ep_local_shards(w: torch.Tensor, p: int) -> list[torch.Tensor]:
    """EP layout: rank r owns experts [r*E/P, (r+1)*E/P]."""
    e = w.shape[0]
    return [w[r * e // p : (r + 1) * e // p].clone() for r in range(p)]


def tp_local_shards(w: torch.Tensor, p: int, shard_dim: int) -> list[torch.Tensor]:
    """TP layout: rank r owns shard [r*D/P, (r+1)*D/P] of ``shard_dim``."""
    d = w.shape[shard_dim]
    out = []
    for r in range(p):
        idx = [slice(None)] * w.dim()
        idx[shard_dim] = slice(r * d // p, (r + 1) * d // p)
        out.append(w[tuple(idx)].clone())
    return out


def simulate_all_to_all(x_shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Simulate dist.all_to_all_single over the first dim."""
    p = len(x_shards)
    return [torch.stack([x_shards[i][r] for i in range(p)], dim=0) for r in range(p)]


def ep_to_tp_input(local_ep: torch.Tensor, p: int, shard_dim: int) -> torch.Tensor:
    e_per = local_ep.shape[0]
    d = local_ep.shape[shard_dim]
    if shard_dim == 1:  # W13 (E/P, 2I, H) -> (P, E/P, 2I/P, H)
        return local_ep.reshape(e_per, p, d // p, -1).transpose(0, 1).contiguous()
    # W2 (E/P, H, I) -> (P, E/P, H, I/P)
    return local_ep.reshape(e_per, -1, p, d // p).permute(2, 0, 1, 3).contiguous()


def tp_to_ep_input(local_tp: torch.Tensor, p: int, shard_dim: int) -> torch.Tensor:
    e = local_tp.shape[0]
    d = local_tp.shape[shard_dim]
    if shard_dim == 1:  # W13 (E, 2I/P, H) -> (P, E/P, 2I/P, H)
        return local_tp.reshape(p, e // p, d, -1)
    # W2 (E, H, I/P) -> (P, E/P, H, I/P)
    return local_tp.reshape(p, e // p, -1, d)


def out_to_tp(o: torch.Tensor) -> torch.Tensor:
    return o.reshape(o.shape[0] * o.shape[1], o.shape[2], o.shape[3])


def out_to_ep(o: torch.Tensor, shard_dim: int, p: int) -> torch.Tensor:
    if shard_dim == 1:  # W13 (P, E/P, 2I/P, H) -> (E/P, 2I, H)
        return o.transpose(0, 1).reshape(o.shape[1], o.shape[2] * p, o.shape[3])
    # W2 (P, E/P, H, I/P) -> (E/P, H, I)
    return o.permute(1, 2, 0, 3).reshape(o.shape[1], o.shape[2], o.shape[3] * p)


def check_all_to_all_math(w: torch.Tensor, p: int, shard_dim: int, label: str) -> None:
    ep = ep_local_shards(w, p)
    tp = tp_local_shards(w, p, shard_dim)

    x = [ep_to_tp_input(s, p, shard_dim) for s in ep]
    for r in range(p):
        assert torch.equal(out_to_tp(simulate_all_to_all(x)[r]), tp[r]), (
            f"{label} EP->TP rank {r}"
        )

    x = [tp_to_ep_input(s, p, shard_dim) for s in tp]
    for r in range(p):
        assert torch.equal(out_to_ep(simulate_all_to_all(x)[r], shard_dim, p), ep[r]), (
            f"{label} TP->EP rank {r}"
        )


def test_layout_identity_w13():
    e, two_i, h, p = 8, 16, 32, 4
    w13 = torch.randn(e, two_i, h)
    assert torch.equal(torch.cat(ep_local_shards(w13, p), dim=0), w13)
    assert torch.equal(torch.cat(tp_local_shards(w13, p, 1), dim=1), w13)


def test_layout_identity_w2():
    e, h, i, p = 6, 24, 12, 3
    w2 = torch.randn(e, h, i)
    assert torch.equal(torch.cat(ep_local_shards(w2, p), dim=0), w2)
    assert torch.equal(torch.cat(tp_local_shards(w2, p, 2), dim=2), w2)


def test_all_to_all_math_w13():
    check_all_to_all_math(torch.randn(8, 16, 32), 4, 1, "W13")


def test_all_to_all_math_w2():
    check_all_to_all_math(torch.randn(6, 24, 12), 3, 2, "W2")
