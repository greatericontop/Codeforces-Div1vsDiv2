"""greatengine (greateric's engine) for rating calcs"""

import math
import time
from dataclasses import dataclass

import numpy as np

from common import Player
from fft import multiply


ROOT2PI = math.sqrt(2 * math.pi)


def sigmoid(x: float) -> float:
    """Sigmoid win/lose probability function in units of rating"""
    return 1 / (1 + math.exp(-x/173.7178))


def sigmoid_g(x: float, opponent_rd: float) -> float:
    """Sigmoid win/lose probability function, adjusted by opponent's RD"""
    g = math.sqrt(173.7178**2 + math.pi/8 * opponent_rd**2)
    return 1 / (1 + math.exp(-x/g))


def npdf(x: float, mu: float, sigma: float) -> float:
    """Normal probability density function"""
    z = (x - mu) / sigma
    return (1 / (sigma * ROOT2PI)) * math.exp(-0.5 * z**2)


@dataclass
class HeliosEngine:
    # Minimum and maximum ratings that we should precompute likelihoods for
    MIN_RATING: float
    MAX_RATING: float
    # The step size in our numerically stored probability density function
    STEP: float
    # Use a fixed RD for all players (lite version), or `None` to enable full calculation
    use_fixed_rd: float = None
    min_rd: float = 0.01
    # new rd = (slow_dev) old_rd + (1-slow_dev) new_rd
    slow_dev: float = 0.0

    def compute_likelihoods(self, ratings: list[tuple[float, float]]) -> dict[float, np.ndarray]:
        """Given the ratings of the field, calculate a dict[rating, likelihood polynomial] using genfunc
        The i'th coefficient of the resulting polynomial is the likelihood of placing i'th (0 based)
        """
        start = time.perf_counter()
        ret = {}
        y = self.MIN_RATING
        while y <= self.MAX_RATING:
            polys = []
            for r, rd in ratings:
                p_lose = sigmoid_g(r - y, rd)
                # win is x^0, lose is x^1
                polys.append(np.array([1-p_lose, p_lose]))
            a = multiply(polys)
            a = np.clip(a, 0, None)
            ret[y] = a
            y += self.STEP
        #print(f'calculated {len(ret)} likelihoods in {time.perf_counter()-start:.3f}s')
        return ret


    def update_player(self, old_rating: float, rd: float, likelihoods: dict[float, np.ndarray], place: int) -> tuple[float, float]:
        """Return the player's new rating and RD. Placing should be 0-based (how many players did you lose to, excluding yourself).
        Time complexity is O(m) where m is (max-min)/step.
        """
        posterior = {}  # this might be kinda slow but it's easier to work with
        y = self.MIN_RATING
        while y <= self.MAX_RATING:
            prior_density = npdf(y, old_rating, rd)
            assert prior_density >= 0, 'Prior density is negative'
            # To avoid having to recalculate likelihoods, we'll just average these two likelihoods of you beating and losing to yourself
            likelihood = float((likelihoods[y][place] + likelihoods[y][place+1]) / 2)
            posterior[y] = prior_density * likelihood
            y += self.STEP
        integ = sum(posterior.values()) * self.STEP
        assert integ >= 0, 'Integral must be nonnegative'
        assert integ > 0, 'Zero integral is bad!'
        if integ <= 1e-15:
            print(f'warning: integral of unnormalized posterior is quite small: {integ}')
        for y in posterior:
            posterior[y] /= integ
        new_density_sum = sum(posterior.values()) * self.STEP
        assert abs(new_density_sum - 1) < 1e-4, f'Posterior density does not integrate to 1 (after multiplying by STEP), got {new_density_sum}'
        # Moment matching
        mu_new = sum(y * density for y, density in posterior.items()) * self.STEP
        var_new = sum((y-mu_new)**2 * density for y, density in posterior.items()) * self.STEP
        return mu_new, math.sqrt(var_new)


    def inference(self, players: list[Player]) -> None:
        """Performs an update, assuming the players are sorted with the winner in index 0.
        Modifies the inputted list of players in-place.
        """
        for player in players:
            if self.use_fixed_rd is not None:
                player.rd = self.use_fixed_rd
        likelihoods = self.compute_likelihoods([(p.rating, p.rd) for p in players])
        for i, player in enumerate(players):
            new_rating, new_rd = self.update_player(player.rating, player.rd, likelihoods, i)
            player.rating = new_rating
            player.rd = max(self.slow_dev*player.rd + (1-self.slow_dev)*new_rd, self.min_rd)
            if self.use_fixed_rd is not None:
                player.rd = self.use_fixed_rd




# my cringe attempt at making an AI model sounding name lol
# assume user ratings range from 0 to 4000
#helios_lite_1_eco = HeliosEngine(MIN_RATING=-1000, MAX_RATING=5000, STEP=10, use_fixed_rd=90)
#helios_1_eco = HeliosEngine(MIN_RATING=-1000, MAX_RATING=5000, STEP=10)
#helios_lite_1_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, use_fixed_rd=90)
#helios_1_high = HeliosEngine(MIN_RATING=-2000, MAX_RATING=6000, STEP=2.5)
#helios_1_max = HeliosEngine(MIN_RATING=-4000, MAX_RATING=8000, STEP=0.05)


helios_1_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5)

# 0 to 30 seems to be good range? would want to benchmark more
helios_1_slowdev15_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, slow_dev=0.15)
#helios_1_slowdev30_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, slow_dev=0.3)
#helios_1_slowdev45_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, slow_dev=0.45)


