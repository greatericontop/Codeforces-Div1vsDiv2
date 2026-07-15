
import math

import numpy as np
from scipy.signal import correlate


MINR = -500
MAXR = 6000
N = MAXR - MINR + 1


def calcperf(contestant_ratings: list[int]) -> dict[int, float]:
    """Given the list of contestant ratings, calculate and return as a dict (perf rating) -> (0-based placing)"""
    onehot = np.zeros(N, dtype=float)
    for r in contestant_ratings:
        if not (MINR <= r <= MAXR):
            raise ValueError(f'Rating {r} is out of bounds [{MINR}, {MAXR}]')
        onehot[r - MINR] += 1
    # The chance you lose to a player with some delta
    window = np.zeros(2*N+1, dtype=float)
    for i in range(2*N+1):
        delta = i - N
        window[i] = 1.0 / (1.0 + math.exp(-delta / 173.7178))

    onehot_padded = np.pad(onehot, (N, N))
    result = correlate(onehot_padded, window, mode='valid', method='fft')
    assert len(result) == N
    return {r: float(result[r - MINR]) for r in range(MINR, MAXR + 1)}


def binsearch(ratingtoplace: dict[int, float], placing: float) -> int:
    """Given :placing:, return the perf."""
    l = MINR
    r = MAXR
    while l < r:
        m = l + (r-l)//2
        if ratingtoplace[m] > placing:
            l = m + 1
        else:
            r = m
    return l
