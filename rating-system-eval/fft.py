"""Polynomial multiplication"""

import numpy as np
from scipy.signal import fftconvolve


def multiply(polys: list[np.ndarray]) -> np.ndarray:
    """Multiply polynomials using FFT. Each polynomial is represented as a[0] + a[1] x + a[2] x^2 + ..."""
    # Convolve pairs of polynomials
    while len(polys) > 1:
        new_polys = []
        for i in range(0, len(polys), 2):
            if i + 1 < len(polys):
                new_polys.append(fftconvolve(polys[i], polys[i+1], mode='full'))
            else:
                new_polys.append(polys[i])
        polys = new_polys
    return polys[0]


if __name__ == '__main__':
    p1 = np.array([2])  # 2
    p2 = np.array([3, 1])  # x + 3
    p3 = np.array([1, 2])  # 2x + 1
    p4 = np.array([0, 1])  # x
    p5 = np.array([0, 1])  # x
    result = multiply([p1, p2, p3, p4, p5])
    print(f'{result=}')
    assert np.allclose(result, np.array([0, 0, 6, 14, 4]))  # 4x^4 + 14x^3 + 6x^2
    print('Test passed')
