
import math
import random
from dataclasses import dataclass, field


@dataclass
class Player:
    real_rating: float
    rating: float
    rd: float
    random_roll: float = field(init=False, default=-1.0)

    def draw(self) -> None:
        # Actually higher numbers are better to not have it exponentially drop off toward 0
        shares = math.exp((self.real_rating - 1200.0) / 173.7178)
        self.random_roll = random.random() ** (1/shares)
