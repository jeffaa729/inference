"""Verify Moebius UMM in-place resharding safety (numpy, no GPU).

Moebius Sec 4.2 uses N+1 slots for N layers with a mode-specific alias offset
(TP: layer i -> slot i; EP: layer i -> slot i+1) so a switch can reshard in
place. This checks the two properties that make it correct:

  1. EP->TP must process layers sequentially (0..N-1), TP->EP in reverse
     (N-1..0); the opposite order loses data.
  2. A full EP->TP->EP roundtrip preserves every layer's data with a single
     scratch slot (the extra slot alternates between slot 0 and slot N).

Run: python swiftep/tests/verify_umm_slots.py
"""


def ep_to_tp(slots, order):
    """Move layer i from slot i+1 (EP) to slot i (TP) in the given order."""
    n = len(slots) - 1  # N layers
    for i in order:
        slots[i] = slots[i + 1]
        slots[i + 1] = None
    return slots


def tp_to_ep(slots, order):
    """Move layer i from slot i (TP) to slot i+1 (EP) in the given order."""
    n = len(slots) - 1
    for i in order:
        slots[i + 1] = slots[i]
        slots[i] = None
    return slots


def main() -> None:
    n = 6
    layers = [f"L{i}" for i in range(n)]

    # Correct orders: EP->TP sequential, TP->EP reverse.
    slots = [None] + layers[:]  # EP layout: layer i in slot i+1
    ep_to_tp(slots, range(n))
    assert slots == layers + [None], "EP->TP sequential must give TP layout"
    tp_to_ep(slots, reversed(range(n)))
    assert slots == [None] + layers, "TP->EP reverse must give EP layout"
    print("roundtrip EP->TP->EP preserves all layers (sequential / reverse)")

    # Wrong order loses data: EP->TP in reverse overwrites the not-yet-read slot.
    slots = [None] + layers[:]
    ep_to_tp(slots, reversed(range(n)))
    assert slots != layers + [None], "reverse EP->TP should NOT be correct"
    assert None in slots[:n] or any(f"L{i}" not in slots for i in range(n)), (
        "reverse order must lose or duplicate some layer"
    )
    print("confirmed: EP->TP reverse order is unsafe (order matters)")

    # The extra slot alternates: after EP->TP it is slot N, after TP->EP slot 0.
    slots = [None] + layers[:]
    ep_to_tp(slots, range(n))
    assert slots[-1] is None and None not in slots[:-1]
    tp_to_ep(slots, reversed(range(n)))
    assert slots[0] is None and None not in slots[1:]
    print("confirmed: single extra slot alternates (slot 0 <-> slot N)")


if __name__ == "__main__":
    main()
