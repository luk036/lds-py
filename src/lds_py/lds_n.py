from functools import cache
from math import cos, sin, sqrt
from typing import List

import numexpr as ne
import numpy as np

from .lds import Circle, Sphere, Vdcorput

PI: float = np.pi
HALF_PI: float = PI / 2.0


class HaltonN:
    """HaltonN sequence generator

    Examples:
        >>> hgen = HaltonN(3, [2, 3, 5])
        >>> hgen.reseed(0)
        >>> for _ in range(10):
        ...     print("{}".format(hgen.pop()))
        ...
        [0.5, 0.3333333333333333, 0.2]
        [0.25, 0.6666666666666666, 0.4]
        [0.75, 0.1111111111111111, 0.6]
        [0.125, 0.4444444444444444, 0.8]
        [0.625, 0.7777777777777777, 0.04]
        [0.375, 0.2222222222222222, 0.24000000000000002]
        [0.875, 0.5555555555555556, 0.44]
        [0.0625, 0.8888888888888888, 0.64]
        [0.5625, 0.037037037037037035, 0.8400000000000001]
        [0.3125, 0.37037037037037035, 0.08]
    """
    vdcs: List[Vdcorput]

    def __init__(self, n: int, base: List[int]):
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
    """CylinN sequence generator

    Examples:
        >>> cgen = CylinN(3, [2, 3, 5, 7])
        >>> cgen.reseed(0)
        >>> for _ in range(10):
        ...     print("{}".format(cgen.pop()))
        ...
        [0.5896942325314937, 0.4702654580212986, -0.565685424949238, -0.33333333333333337, 0.0]
        [0.7799423297454589, -0.17801674716505145, -0.16329931618554513, 0.2886751345948128, -0.5]
        [0.2314046608626977, -0.4805167295479567, 0.10886621079036342, -0.6735753140545634, 0.5]
        [-0.2281680726408047, -0.47379588485330215, 0.39440531887330776, -0.07349309197401645, -0.75]
        [-0.30761340326604825, -0.07021075193042121, -0.7406703670274604, 0.5379143536399188, 0.25]
        [-0.5376414074929536, 0.42875471523691105, -0.4186397726676949, -0.537914353639919, -0.25]
        [0.08345163471779245, 0.6472343994819842, -0.07888106377466154, 0.07349309197401645, 0.75]
        [0.2498031899655053, 0.15143216176357996, 0.08520128672302588, 0.3765400475479433, -0.875]
        [0.25785911461779826, -0.09489453780287567, 0.2548250428610497, -0.9186636496752051, 0.125]
        [0.1530772608880413, -0.46104432024542363, -0.7520753180038667, -0.24033976578550764, -0.375]
    """
    vdc: Vdcorput

    def __init__(self, n: int, base: List[int]):
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

    def reseed(self, seed: int):
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.c_gen.reseed(seed)


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
    """Sphere3 sequence generator

    Examples:
        >>> sgen = Sphere3([2, 3, 5])
        >>> sgen.reseed(0)
        >>> for _ in range(10):
        ...     print("{}".format(sgen.pop()))
        ...
        [0.8966646826186098, 0.2913440162992141, -0.33333333333333337, 6.123233995736766e-17]
        [0.5069371683663506, -0.697739153354296, 0.30492319090118075, 0.4039760251002259]
        [-0.33795811224423367, -0.4651594355695309, -0.7114874454360887, -0.4039760251002258]
        [-0.7303800100566173, 0.23731485100218302, -0.08586131969610163, 0.634708229175856]
        [0.20270132011818542, 0.7894695182151528, 0.544595994329299, -0.19764932985457223]
        [0.8134682824136722, 0.051179086309454114, -0.5445959943292992, 0.19764932985457237]
        [0.2827075046384319, -0.7140376491597653, 0.08586131969610163, -0.6347082291758559]
        [-0.30699987732880385, -0.2539723859311107, 0.49303885423959726, 0.7734092000452695]
        [-0.3173613919107008, 0.2014036636020021, -0.9214383378738055, -0.09833463636253847]
        [0.443988803653532, 0.8076126357141739, -0.24739479977418324, 0.29905114263249766]
    """
    vdc: Vdcorput
    sphere2: Sphere

    def __init__(self, base: List[int]):
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
    """SphereN sequence generator

    Examples:
        >>> sgen = SphereN(3, [2, 3, 5, 7])
        >>> sgen.reseed(0)
        >>> for _ in range(10):
        ...     print("{}".format(sgen.pop()))
        ...
        [0.6031153874276115, 0.4809684718990214, -0.5785601510223212, 0.2649326520763179, 6.123233995736766e-17]
        [0.8637599808754701, -0.19714757907418667, -0.18084851769929866, -0.24844094775703415, 0.34730740940375543]
        [0.29829795560026834, -0.6194220873577418, 0.14033670709848772, 0.6221075304094009, -0.34730740940375554]
        [-0.2869679379505121, -0.5958950632844812, 0.49604521686066244, 0.07251736176256243, 0.5578878482403709]
        [-0.33592237258061136, -0.07667208944318125, -0.8088325942573712, -0.4458201230678714, -0.16826111444676858]
        [-0.587119336430247, 0.46821204690141727, -0.45716624892807933, 0.44582012306787167, 0.16826111444676847]
        [0.10495746953722358, 0.8140294075340622, -0.0992090433721325, -0.07251736176256235, -0.5578878482403704]
        [0.44138662544233553, 0.26757116621887106, 0.15054534906145225, -0.4766871927197732, 0.6954774878181234]
        [0.37066313847851823, -0.1364074613325864, 0.3663017702120042, 0.8383599596078609, -0.08353103334636607]
        [0.16178258849951696, -0.4872633800055925, -0.7948449755855819, 0.19826109691937244, 0.25556759340457186]
    """
    vdc: Vdcorput
    n: int

    def __init__(self, n: int, base: List[int]):
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

    def reseed(self, seed: int):
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.s_gen.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
