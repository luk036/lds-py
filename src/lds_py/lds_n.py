from functools import cache
from math import cos, sin, sqrt
from typing import List

import numexpr as ne
import numpy as np

from .lds import PRIME_TABLE, Circle, Sphere, Vdcorput

PI: float = np.pi
HALF_PI: float = PI / 2.0


class HaltonN:
    vdcs: List[Vdcorput]

    def __init__(self, n: int, base: List[int] = PRIME_TABLE):
        """_summary_

        Args:
            base (List[int]): _description_
        """
        self.vdcs = [Vdcorput(base[i]) for i in range(n)]

    def pop(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        return [vdc.pop() for vdc in self.vdcs]

    def reseed(self, seed: int):
        """_summary_

        Args:
            seed (int): _description_
        """
        for vdc in self.vdcs:
            vdc.reseed(seed)


# CylinVariant = Union[Circle, CylinN]


class CylinN:
    vdc: Vdcorput

    def __init__(self, n: int, base: List[int] = PRIME_TABLE):
        """_summary_

        Args:
            base (List[int]): _description_
        """
        assert n >= 1
        self.vdc = Vdcorput(base[0])
        self.c_gen = Circle(base[1]) if n == 1 else CylinN(n - 1, base[1:])

    def pop(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        cosphi = 2.0 * self.vdc.pop() - 1.0  # map to [-1, 1]
        sinphi = sqrt(1.0 - cosphi * cosphi)
        return [xi * sinphi for xi in self.c_gen.pop()] + [cosphi]


X: np.ndarray = np.linspace(0.0, PI, 300)
NEG_COSINE: np.ndarray = -np.cos(X)
SINE: np.ndarray = np.sin(X)


@cache
def get_tp(n: int) -> np.ndarray:
    """_summary_

    Returns:
        np.ndarray: _description_
    """
    if n == 0:
        return X
    if n == 1:
        return NEG_COSINE;
    tp_minus2 = get_tp(n - 2)  # NOQA
    return ne.evaluate("((n - 1) * tp_minus2 + NEG_COSINE * SINE**(n - 1)) / n")
 

class Sphere3:
    vdc: Vdcorput
    sphere2: Sphere

    def __init__(self, base: List[int] = PRIME_TABLE):
        """_summary_

        Args:
            base (List[int]): _description_
        """
        self.vdc = Vdcorput(base[0])
        self.sphere2 = Sphere(base[1:3])

    def reseed(self, seed: int):
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.sphere2.reseed(seed)

    def pop(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        ti = HALF_PI * self.vdc.pop()  # map to [0, pi/2]
        xi = np.interp(ti, get_tp(2), X)
        cosxi = cos(xi)
        sinxi = sin(xi)
        return [sinxi * s for s in self.sphere2.pop()] + [cosxi]


# SphereVaiant = Union[Sphere3, SphereN]

class SphereN:
    vdc: Vdcorput
    n: int

    def __init__(self, n: int, base: List[int] = PRIME_TABLE):
        """_summary_

        Args:
            base (List[int]): _description_
        """
        assert n >= 2
        self.vdc = Vdcorput(base[0])
        s_gen = Sphere(base[1:3]) if n == 2 else SphereN(n - 1, base[1:])
        self.s_gen = s_gen
        self.n = n
        tp = get_tp(n)
        self.range = tp[-1] - tp[0]

    def pop(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        vd = self.vdc.pop()
        tp = get_tp(self.n)
        ti = tp[0] + self.range * vd  # map to [t0, tm-1]
        xi = np.interp(ti, tp, X)
        sinphi = sin(xi)
        return [xi * sinphi for xi in self.s_gen.pop()] + [cos(xi)]
