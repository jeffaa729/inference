"""Standalone numpy verification of the Moebius resharding slice math.

Mirrors ``tests/test_resharding.py`` and the IPC direct-write path of
``benchmarks/resharding_benchmark.py``, but runs on numpy (no torch/GPU), so both
the all-to-all reshape/permute logic AND the direct-write narrow/slice logic are
checked locally before touching the 8xA100 cluster.

Run: python swiftep/tests/verify_resharding_numpy.py

Covers both weight tensors:
  * W13 (gate+up, shard the 2I dim = dim 1 of (E, 2I, H))
  * W2  (down,      shard the  I dim = dim 2 of (E, H, I))
"""

import numpy as np


def ep_local(w, p):
    e = w.shape[0]
    return [w[r * e // p:(r + 1) * e // p] for r in range(p)]


def tp_local(w, p, d):
    dlen = w.shape[d]
    return [np.take(w, np.arange(r * dlen // p, (r + 1) * dlen // p), axis=d) for r in range(p)]


def simulate_all_to_all(x_shards):
    p = len(x_shards)
    return [np.stack([x_shards[i][r] for i in range(p)], axis=0) for r in range(p)]


def ep_to_tp_input(s, p, d):
    e_per, dlen = s.shape[0], s.shape[d]
    if d == 1:  # W13 (E/P, 2I, H) -> (P, E/P, 2I/P, H)
        return s.reshape(e_per, p, dlen // p, -1).transpose(1, 0, 2, 3)
    return s.reshape(e_per, -1, p, dlen // p).transpose(2, 0, 1, 3)  # W2


def tp_to_ep_input(s, p, d):
    e, dlen = s.shape[0], s.shape[d]
    if d == 1:  # W13 (E, 2I/P, H) -> (P, E/P, 2I/P, H)
        return s.reshape(p, e // p, dlen, -1)
    return s.reshape(p, e // p, -1, dlen)  # W2


def out_to_tp(o):
    return o.reshape(o.shape[0] * o.shape[1], o.shape[2], o.shape[3])


def out_to_ep(o, d, p):
    if d == 1:
        return o.transpose(1, 0, 2, 3).reshape(o.shape[1], o.shape[2] * p, o.shape[3])
    return o.transpose(1, 2, 0, 3).reshape(o.shape[1], o.shape[2], o.shape[3] * p)


def check_nccl(w, p, d, label):
    ep, tp = ep_local(w, p), tp_local(w, p, d)
    x = [ep_to_tp_input(s, p, d) for s in ep]
    for r in range(p):
        assert np.array_equal(out_to_tp(simulate_all_to_all(x)[r]), tp[r]), f"{label} NCCL EP->TP rank {r}"
    x = [tp_to_ep_input(s, p, d) for s in tp]
    for r in range(p):
        assert np.array_equal(out_to_ep(simulate_all_to_all(x)[r], d, p), ep[r]), f"{label} NCCL TP->EP rank {r}"
    print(f"{label}: NCCL math OK")


def check_ipc(w, p, d, label):
    ep, tp = ep_local(w, p), tp_local(w, p, d)
    e_per = w.shape[0] // p
    dlen = w.shape[d]

    # EP -> TP direct write: rank r writes its slice into each dest's TP buffer.
    tp_shape = list(ep[0].shape)
    tp_shape[0], tp_shape[d] = w.shape[0], dlen // p
    dst_tp = [np.zeros(tp_shape) for _ in range(p)]
    for r in range(p):
        for dest in range(p):
            src = np.take(ep[r], np.arange(dest * (dlen // p), (dest + 1) * (dlen // p)), axis=d)
            idx = [slice(None)] * len(tp_shape)
            idx[0] = slice(r * e_per, (r + 1) * e_per)
            dst_tp[dest][tuple(idx)] = src
    for r in range(p):
        assert np.array_equal(dst_tp[r], tp[r]), f"{label} IPC EP->TP rank {r}"

    # TP -> EP direct write: rank r writes its TP slice of dest's experts.
    ep_shape = list(tp[0].shape)
    ep_shape[0], ep_shape[d] = e_per, dlen
    dst_ep = [np.zeros(ep_shape) for _ in range(p)]
    for r in range(p):
        for dest in range(p):
            idx = [slice(None)] * len(ep_shape)
            idx[d] = slice(r * (dlen // p), (r + 1) * (dlen // p))
            dst_ep[dest][tuple(idx)] = tp[r][dest * e_per:(dest + 1) * e_per]
    for r in range(p):
        assert np.array_equal(dst_ep[r], ep[r]), f"{label} IPC TP->EP rank {r}"
    print(f"{label}: IPC direct-write math OK")


rng = np.random.default_rng(0)
for w, p, d, label in [
    (rng.standard_normal((8, 16, 32)), 4, 1, "W13 (2I sharded)"),
    (rng.standard_normal((6, 24, 12)), 3, 2, "W2  (I sharded)"),
]:
    check_nccl(w, p, d, label)
    check_ipc(w, p, d, label)
print("all resharding math checks passed")
