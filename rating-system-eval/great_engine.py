"""greatengine (greateric's engine) for rating calcs"""

import math
import time
from dataclasses import dataclass

import numpy as np

from common import Player
from fft import multiply


ROOT2PI = math.sqrt(2 * math.pi)


def sigmoid(x: float) -> float:
    """Sigmoid function in units of rating"""
    return 1 / (1 + math.exp(-x/173.7178))


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

    def compute_likelihoods(self, ratings: list[float]) -> dict[float, np.ndarray]:
        """Given the ratings of the field, calculate a dict[rating, likelihood polynomial] using genfunc
        The i'th coefficient of the resulting polynomial is the likelihood of placing i'th (0 based)
        """
        start = time.perf_counter()
        ret = {}
        y = self.MIN_RATING
        while y <= self.MAX_RATING:
            polys = []
            for r in ratings:
                p_lose = sigmoid(r - y)
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
        integ = max(sum(posterior.values()) * self.STEP, 0.000001)
        for y in posterior:
            posterior[y] /= integ
        # Moment matching
        mu_new = sum(y * density for y, density in posterior.items()) * self.STEP
        var_new = sum((y-mu_new)**2 * density for y, density in posterior.items()) * self.STEP
        return mu_new, math.sqrt(var_new)


    def inference(self, players: list[Player]) -> None:
        """Performs an update, assuming the players are sorted with the winner in index 0.
        Modifies the inputted list of players in-place.
        """
        likelihoods = self.compute_likelihoods([p.rating for p in players])
        for i, player in enumerate(players):
            new_rating, new_rd = self.update_player(player.rating, self.use_fixed_rd if self.use_fixed_rd is not None else player.rd, likelihoods, i)
            player.rating = new_rating
            player.rd = max(self.slow_dev*player.rd + (1-self.slow_dev)*new_rd, self.min_rd)


# my cringe attempt at making an AI model sounding name lol
# assume user ratings range from 0 to 4000
#helios_lite_1_eco = HeliosEngine(MIN_RATING=-1000, MAX_RATING=5000, STEP=10, use_fixed_rd=90)
helios_1_eco = HeliosEngine(MIN_RATING=-1000, MAX_RATING=5000, STEP=10)
#helios_lite_1_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, use_fixed_rd=90)
helios_1_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5)
#helios_1_hce70_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, min_rd=70)
#helios_1_hce60_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, min_rd=60)
#helios_1_hce50_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, min_rd=50)
#helios_1_hce50slowdev15_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, min_rd=50, slow_dev=0.15)
#helios_1_hce50slowdev30_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, min_rd=50, slow_dev=0.3)
#helios_1_hce50slowdev45_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, min_rd=50, slow_dev=0.45)
#helios_1_hce40_medium = HeliosEngine(MIN_RATING=-1500, MAX_RATING=5500, STEP=5, min_rd=40)
helios_1_high = HeliosEngine(MIN_RATING=-2000, MAX_RATING=6000, STEP=2.5)
#helios_1_max = HeliosEngine(MIN_RATING=-4000, MAX_RATING=8000, STEP=0.05)







# For testing/tuning
if __name__ == '__main__':
    def main():
        combos = [
            (600, 2800, 2, '400 dynamic range, low precision'),
            (400, 3000, 2, '600 dynamic range, low precision'),
            (200, 3200, 2, '800 dynamic range, low precision'),
            (200, 3200, 1, '800 dynamic range, medium precision'),
            (-1000, 4400, 0.02, 'ground truth'),
        ]
        for combo in combos:
            print(combo[3])
            engine = HeliosEngine(MIN_RATING=combo[0], MAX_RATING=combo[1], STEP=combo[2])
            likelihoods = engine.compute_likelihoods([1100, 1130, 1200])
            players = [
                (1000, 150),
                (1300, 150),
                (2400, 150),
                (2400, 350),
            ]
            for player in players:
                new_rating, new_rd = engine.update_player(player[0], player[1], likelihoods, 0)
                print(f'Player ({player[0]}, rd={player[1]})  ->  ({new_rating}, rd={new_rd}) from winning')
            print()
    main()