from __future__ import annotations

import numpy as np


def raised_cosine_taper(n: int, p: float = 0.01) -> np.ndarray:
    m = int(np.floor(p * n))
    if m == 0:
        return np.ones(n)
    w = np.ones(n)
    t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, m)))
    w[:m] = t
    w[-m:] = t[::-1]
    return w


def traffic_band_envelope(freqs: np.ndarray, f1: float = 1.0, f2: float = 49.0, ramp: float = 1.0) -> np.ndarray:
    bp = np.zeros_like(freqs, dtype=float)
    mid = (freqs >= f1) & (freqs <= f2)
    bp[mid] = 1.0
    lo = (freqs >= (f1 - ramp)) & (freqs < f1)
    hi = (freqs > f2) & (freqs <= (f2 + ramp))
    bp[lo] = 0.5 * (1 - np.cos(np.pi * (freqs[lo] - (f1 - ramp)) / ramp))
    bp[hi] = 0.5 * (1 + np.cos(np.pi * (freqs[hi] - f2) / ramp))
    return bp






