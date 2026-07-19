"""greatengine (greateric's engine) for rating calcs"""

import math
import time
from dataclasses import dataclass

import numba
import numpy as np
from scipy.signal import correlate

from common import Player


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
    They are returned in base 10 log space.
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
            # TODO: doing these log2 and np.exp2() SIMD may be faster
            prefix_pro[it][i] = prefix_pro[it][i-1] + math.log10(1.0 - likelihoods[i])
        suffix_pro[it][-1] = math.log10(likelihoods[-1])
        for i in range(n-2, -1, -1):
            suffix_pro[it][i] = suffix_pro[it][i+1] + math.log10(likelihoods[i])

    return prefix_pro, suffix_pro


@numba.jit(fastmath=True)
def _update_player_initial(rating: float, rd: float, place: int,
                           prefix_products: np.ndarray, suffix_products: np.ndarray,
                           MIN_RATING: float, STEP: float) -> tuple[np.ndarray, np.ndarray]:
    """Return filter and padded perf likelihoods"""
    m, n = prefix_products.shape
    # For each perf (m of them), calculate the actual likelihood (prefix sum of loss, suffix sum of win)
    perf_likelihoods = np.empty(m, dtype=np.float64)
    for it in range(m):
        prefix = prefix_products[it][place - 1] if place > 0 else 0.0  # 0.0 is no-op/100% in log space
        suffix = suffix_products[it][place + 1] if place < n - 1 else 0.0
        perf_likelihoods[it] = prefix + suffix
        if perf_likelihoods[it] == -float('inf'):
            perf_likelihoods[it] = -1e100
    #print('raw log space likelihoods: ', perf_likelihoods)
    # Unlog, also move the max value to 10^100 to avoid underflow (values are probably ~= -n)
    perf_likelihoods += 100 - np.max(perf_likelihoods)
    perf_likelihoods = 10 ** perf_likelihoods
    #print('perf likelihoods: ', perf_likelihoods)

    # For true rating `x`, calculate E[likelihood] via convolution
    # TODO: maybe cut off filter sooner
    filter = np.empty(2*m+1, dtype=np.float64)
    for i in range(0, 2*m+1):
        r_diff = (i - m) * STEP
        filter[i] = npdf(r_diff / BETA)
    padded = np.zeros(3*m, dtype=np.float64)
    padded[m:2*m] = perf_likelihoods
    return filter, padded


@numba.jit(fastmath=True)
def _update_player_posterior(filter: np.ndarray, truerating_likelihoods: np.ndarray, rating: float, rd: float, MIN_RATING: float, STEP: float) -> tuple[float, float]:
    """Perform MAP posterior update step with true rating likelihoods. Return the results"""
    m = truerating_likelihoods.shape[0]
    # Create the new distribution
    # Prior(=probability of x under rating, rd) * Likelihood(=probability of x from truerating_likelihood)
    posterior = np.empty(m, dtype=np.float64)
    x = MIN_RATING + STEP * np.arange(m, dtype=np.float64)  # x[i] = rating value for i value
    #print('x: ', x)
    for i in range(m):
        prior_density = npdf((x[i] - rating) / rd)
        likelihood = truerating_likelihoods[i]
        posterior[i] = prior_density * likelihood
    posterior /= STEP * np.sum(posterior)  # Normalize to 1 integral (which is sum of 1/STEP)
    #print('posterior: ', list(posterior))
    mu_new = np.sum(x * posterior) * STEP
    var_new = np.sum((x-mu_new)**2 * posterior) * STEP
    return mu_new, math.sqrt(var_new)


def _update_player(rating: float, rd: float, place: int,
                   prefix_products: np.ndarray, suffix_products: np.ndarray,
                   MIN_RATING: float, STEP: float
                   ) -> tuple[float, float]:
    """
    Return new rating and RD of player.
    Time complexity: O(m log m) per player (due to FFT)
    """
    filter, padded = _update_player_initial(rating, rd, place, prefix_products, suffix_products, MIN_RATING, STEP)
    truerating_likelihoods = correlate(padded, filter, mode='valid', method='auto')
    return _update_player_posterior(filter, truerating_likelihoods, rating, rd, MIN_RATING, STEP)


@dataclass
class Helios2:
    MIN_RATING: float
    MAX_RATING: float
    STEP: float


    def compute_likelihoods(self, ratings: list[float], rds: list[float]) -> tuple[np.ndarray, np.ndarray]:
        return _compute_likelihoods(ratings, rds, self.MIN_RATING, self.MAX_RATING, self.STEP)


    def inference(self, players: list[Player]) -> None:
        ratings = [p.rating for p in players]
        rds = [p.rd for p in players]
        prefix_products, suffix_products = self.compute_likelihoods(ratings, rds)
        for place, player in enumerate(players):
            new_rating, new_rd = _update_player(player.rating, player.rd, place,
                                                prefix_products, suffix_products,
                                                self.MIN_RATING, self.STEP)
            player.rating = new_rating
            player.rd = new_rd


helios_2_eco = Helios2(MIN_RATING=-1000, MAX_RATING=5000, STEP=10)
#helios_2_eco_hdr = Helios2(MIN_RATING=-2000, MAX_RATING=6000, STEP=10)
helios_2_medium = Helios2(MIN_RATING=-1000, MAX_RATING=5000, STEP=5)
#helios_2_medium_hdr = Helios2(MIN_RATING=-2000, MAX_RATING=6000, STEP=5)
#helios_2_high = Helios2(MIN_RATING=-1000, MAX_RATING=5000, STEP=2.5)
#helios_2_high_hdr = Helios2(MIN_RATING=-2000, MAX_RATING=6000, STEP=2.5)
