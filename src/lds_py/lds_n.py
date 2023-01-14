from functools import lru_cache
from math import cos, sin, sqrt
from typing import List

# import numexpr as ne
import numpy as np

from .lds import Circle, Sphere, Vdcorput

PI: float = np.pi
HALF_PI: float = PI / 2.0


class HaltonN:
    """HaltonN sequence generator

    Examples:
        >>> hgen = HaltonN(3, [2, 3, 5])
        >>> hgen.reseed(0)
        >>> for _ in range(2):
        ...     print(hgen.pop())
        ...
        [0.5, 0.3333333333333333, 0.2]
        [0.25, 0.6666666666666666, 0.4]
    """

    vdcs: List[Vdcorput]

    def __init__(self, n: int, base: List[int]) -> None:
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

    def reseed(self, seed: int) -> None:
        """_summary_

        Args:
            seed (int): _description_
        """
        for vdc in self.vdcs:
            vdc.reseed(seed)


# CylinVariant = Union[Circle, CylinN]


class CylinN(Circle):
    """CylinN sequence generator

    Examples:
        >>> cgen = CylinN(3, [2, 3, 5, 7])
        >>> cgen.reseed(0)
        >>> for _ in range(1):
        ...     print(cgen.pop())
        ...
        [0.5896942325314937, 0.4702654580212986, -0.565685424949238, -0.33333333333333337, 0.0]
    """

    vdc: Vdcorput
    c_gen: Circle

    def __init__(self, n: int, base: List[int]) -> None:
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

    def reseed(self, seed: int) -> None:
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.c_gen.reseed(seed)


X: np.ndarray = np.linspace(0.0, PI, 300)
NEG_COSINE: np.ndarray = -np.cos(X)
SINE: np.ndarray = np.sin(X)


@lru_cache
def get_tp(n: int) -> np.ndarray:
    """_summary_

    Returns:
        np.ndarray: _description_
    """
    if n == 0:
        return X
    if n == 1:
        return NEG_COSINE
    tp_minus2 = get_tp(n - 2)  # NOQA
    # return ne.evaluate("((n - 1) * tp_minus2 + NEG_COSINE * SINE**(n - 1)) / n")
    return ((n - 1) * tp_minus2 + NEG_COSINE * SINE ** (n - 1)) / n


class Sphere3:
    """Sphere3 sequence generator

    Examples:
        >>> sgen = Sphere3([2, 3, 5])
        >>> sgen.reseed(0)
        >>> for _ in range(1):
        ...     print(sgen.pop())
        ...
        [0.8966646826186098, 0.2913440162992141, -0.33333333333333337, 6.123233995736766e-17]
    """

    vdc: Vdcorput
    sphere2: Sphere

    def __init__(self, base: List[int]) -> None:
        """_summary_

        Args:
            base (List[int]): _description_
        """
        self.vdc = Vdcorput(base[0])
        self.sphere2 = Sphere(base[1:3])

    def reseed(self, seed: int) -> None:
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


class SphereN(Sphere):
    """SphereN sequence generator

    Examples:
        >>> sgen = SphereN(3, [2, 3, 5, 7])
        >>> sgen.reseed(0)
        >>> for _ in range(1):
        ...     print(sgen.pop())
        ...
        [0.6031153874276115, 0.4809684718990214, -0.5785601510223212, 0.2649326520763179, 6.123233995736766e-17]
    """

    vdc: Vdcorput
    s_gen: Sphere
    n: int

    def __init__(self, n: int, base: List[int]) -> None:
        """_summary_

        Args:
            base (List[int]): _description_
        """
        assert n >= 2
        self.vdc = Vdcorput(base[0])
        self.s_gen = Sphere(base[1:3]) if n == 2 else SphereN(n - 1, base[1:])
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

    def reseed(self, seed: int) -> None:
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.s_gen.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
