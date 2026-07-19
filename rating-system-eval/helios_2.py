"""greatengine (greateric's engine) for rating calcs"""

import math
import time
from dataclasses import dataclass

import numba
import numpy as np

from common import Player
from fft import multiply


ROOT2BETA = 299.6
BETA = ROOT2BETA / math.sqrt(2)


@numba.jit(numba.float64(numba.float64))
def ncdf(z: float) -> float:
    """CDF of standard normal"""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@numba.jit(numba.float64(numba.float64))
def npdf(z: float) -> float:
    """PDF of standard normal"""
    return math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)


@numba.jit
def _compute_likelihoods(ratings: list[float], rds: list[float],
                        MIN_RATING: float, MAX_RATING: float, STEP: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return prefix and suffix products of likelihoods.
    Time complexity: O(nm) where n is the number of players and m is the number of steps.
    """
    n = len(ratings)
    m = int((MAX_RATING - MIN_RATING) / STEP) + 1
    prefix_pro = np.empty((m, n), dtype=np.float64)
    suffix_pro = np.empty((m, n), dtype=np.float64)
    for it in range(m):
        y = MIN_RATING + STEP * it
        # Win likelihoods
        likelihoods = np.empty(n, dtype=np.float64)
        for i in range(n):
            # Opponent's perf is sampled from N(rating, sqrt(rd^2 + BETA^2))
            # Calculate probability that is lower than ours
            z = (y - ratings[i]) / math.sqrt(rds[i]**2 + BETA**2)
            likelihoods[i] = ncdf(z)

        prefix_pro[it][0] = math.log10(1.0 - likelihoods[0])
        for i in range(1, n):
            # These are in log space to maintain precision
            prefix_pro[it][i] = prefix_pro[it][i-1] + math.log10(1.0 - likelihoods[i])
        suffix_pro[it][-1] = math.log10(likelihoods[-1])
        for i in range(n-2, -1, -1):
            suffix_pro[it][i] = suffix_pro[it][i+1] + math.log10(likelihoods[i])

    return prefix_pro, suffix_pro


@numba.jit
def _update_player(rating: float, rd: float, place: int,
                   prefix_products: np.ndarray, suffix_products: np.ndarray,
                   MIN_RATING: float, STEP: float
                   ) -> tuple[float, float]:
    """
    Return new rating and RD of player.
    Time complexity: O(m log m) per player (due to 2 FFTs)
    """
    m, n = prefix_products.shape
    # For each perf (m of them), calculate the actual likelihood (prefix sum of loss, suffix sum of win)
    perf_likelihoods = np.empty(m, dtype=np.float64)
    for it in range(m):
        prefix = prefix_products[it][place - 1] if place > 0 else 0.0  # 0.0 is no-op/100% in log space
        suffix = suffix_products[it][place + 1] if place < n - 1 else 0.0
        perf_likelihoods[it] = prefix + suffix
    # Normalize and un-log
    perf_likelihoods -= np.mean(perf_likelihoods)
    perf_likelihoods = 10 ** perf_likelihoods
    print(perf_likelihoods)

    # For true rating `x`, calculate E[likelihood] via convolution
    filter = np.empty(2 * m + 1, dtype=np.float64)
    for i in range(0, 2 * m):
        r_diff = (i - m) * STEP
        filter[i] = npdf(r_diff / BETA)
    padded = np.zeros(3*m, dtype=np.float64)
    padded[m:2*m] = perf_likelihoods
    truerating_likelihoods = np.correlate(padded, filter, mode='valid')
    assert truerating_likelihoods.shape == (m,)
    print(truerating_likelihoods)

    # Create the new distribution
    # Prior(=probability of x under rating, rd) * Likelihood(=probability of x from truerating_likelihood)
    posterior = np.empty(m, dtype=np.float64)
    x = MIN_RATING + STEP * np.arange(m, dtype=np.float64)  # x[i] = rating value for i value
    print('x: ', x)
    for i in range(m):
        prior_density = npdf((x[i] - rating) / rd)
        likelihood = truerating_likelihoods[i]
        posterior[i] = prior_density * likelihood
    posterior /= STEP * np.sum(posterior)  # Normalize to 1 integral (which is sum of 1/STEP)
    print('posterior: ', list(posterior))
    mu_new = np.sum(x * posterior) * STEP
    var_new = np.sum((x-mu_new)**2 * posterior) * STEP
    return mu_new, math.sqrt(var_new)


@dataclass
class Helios2:
    MIN_RATING: float
    MAX_RATING: float
    STEP: float


    def compute_likelihoods(self, ratings: list[float], rds: list[float]) -> tuple[np.ndarray, np.ndarray]:
        return _compute_likelihoods(ratings, rds, self.MIN_RATING, self.MAX_RATING, self.STEP)


    # update_player





    def inference(self, players: list[Player]) -> None:
        pass


