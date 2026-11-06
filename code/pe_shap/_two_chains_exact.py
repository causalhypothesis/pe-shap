"""Exact values for the two-chain design by enumeration over the 16 node configurations."""
import itertools, numpy as np

P = {"A1": lambda T: 0.2 + 0.6 * T, "A2": lambda a1: 0.2 + 0.6 * a1,
     "B1": lambda T: 0.3 + 0.4 * T, "B2": lambda b1: 0.1 + 0.5 * b1}


def law(T, pins):
    """P(A1,A2,B1,B2 | do(T), pins) as {config: prob}."""
    out = {}
    for a1, a2, b1, b2 in itertools.product((0, 1), repeat=4):
        p = 1.0
        for name, val, parent in (("A1", a1, T), ("A2", a2, a1), ("B1", b1, T), ("B2", b2, b1)):
            if name in pins:
                if pins[name] != val: p = 0.0
            else:
                q = P[name](parent); p *= q if val else 1 - q
        out[(a1, a2, b1, b2)] = p
    return out


def mean(f, T, pins):
    return sum(p * f(T, *cfg) for cfg, p in law(T, pins).items())


def pe_values(f, pins_by_path):
    """Game with paths A={A1,A2}, B={B1,B2} as players at the given pins (dict of dicts)."""
    pinsA, pinsB = pins_by_path["A"], pins_by_path["B"]
    v = {}
    for S in [(), ("A",), ("B",), ("A", "B")]:
        pins = {}
        if "A" in S: pins.update(pinsA)
        if "B" in S: pins.update(pinsB)
        v[S] = mean(f, 1, pins) - mean(f, 0, pins)
    ate = v[()]
    v = {S: 0.0 if S == () else ate - x for S, x in v.items()}
    phi = {"A": 0.5 * (v[("A",)] + v[("A", "B")] - v[("B",)]), "B": 0.5 * (v[("B",)] + v[("A", "B")] - v[("A",)])}
    return v, phi
