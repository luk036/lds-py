from math import cos, pi, sin, sqrt
from typing import List

TWO_PI = 2.0 * pi


def vdc(k: int, base: int) -> float:
    vdc = 0.0
    denom = 1.0
    while k != 0:
        denom *= base
        remainder = k % base
        k //= base
        vdc += remainder / denom
    return vdc


class Vdcorput:
    count: int
    base: int

    def __init__(self, base: int = 2):
        self.count = 0.0
        self.base = base

    def pop(self) -> float:
        self.count += 1
        return vdc(self.count, self.base)

    #[allow(dead_code)]
    def reseed(self, seed: int):
        self.count = seed


class Halton:
    vdc0: Vdcorput
    vdc1: Vdcorput

    def __init__(self, base: List[int]):
        self.vdc0 = Vdcorput(base[0])
        self.vdc1 = Vdcorput(base[1])
    
    def pop(self) -> List[float]:
        return [self.vdc0.pop(), self.vdc1.pop()]

    #[allow(dead_code)]
    def reseed(self, seed: int):
        self.vdc0.reseed(seed)
        self.vdc1.reseed(seed)


class Circle:
    vdc: Vdcorput

    def __init__(self, base: int):
        self.vdc = Vdcorput(base)
    
    def pop(self) -> List[float]:
        theta = self.vdc.pop() * TWO_PI # map to [0, 2*pi]
        return [sin(theta), cos(theta)]

    #[allow(dead_code)]
    def reseed(self, seed: int):
        self.vdc.reseed(seed)


class Sphere:
    vdc: Vdcorput
    cirgen: Circle

    def __init__(self, base: List[int]):
        self.vdc = Vdcorput(base[0])
        self.cirgen = Circle(base[1])
    
    def pop(self) -> List[float]:
        cosphi = 2.0 * self.vdc.pop() - 1.0 # map to [-1, 1]
        sinphi = sqrt(1.0 - cosphi * cosphi)
        [c, s] = self.cirgen.pop()
        return [sinphi * c, sinphi * s, cosphi]

    #[allow(dead_code)]
    def reseed(self, seed: int):
        self.cirgen.reseed(seed)
        self.vdc.reseed(seed)


# S(3) sequence generator by Hopf
class Sphere3Hopf:
    vdc0: Vdcorput
    vdc1: Vdcorput
    vdc2: Vdcorput

    def __init__(self, base: List[int]):
        self.vdc0 = Vdcorput(base[0])
        self.vdc1 = Vdcorput(base[1])
        self.vdc2 = Vdcorput(base[2])
    
    def pop(self) -> List[float]:
        phi = self.vdc0.pop() * TWO_PI # map to [0, 2*pi]
        psy = self.vdc1.pop() * TWO_PI # map to [0, 2*pi]
        vd = self.vdc2.pop()
        cos_eta = sqrt(vd)
        sin_eta = sqrt(1.0 - vd)
        return [
            cos_eta * cos(psy),
            cos_eta * sin(psy),
            sin_eta * cos(phi + psy),
            sin_eta * cos(phi + psy),
        ]

    #[allow(dead_code)]
    def reseed(self, seed: int):
        self.vdc0.reseed(seed)
        self.vdc1.reseed(seed)
        self.vdc2.reseed(seed)


if __name__ ==  "__main__":
    base = [2, 3, 5, 7];

    vgen = Vdcorput(2);
    for _ in range(10):
        print("{}".format(vgen.pop()))

    cgen = Circle(2);
    for _ in range(10):
        print("{}".format(cgen.pop()))

    hgen = Halton(base);
    for _ in range(10):
        print("{}".format(hgen.pop()))

    sgen = Sphere(base);
    for _ in range(10):
        print("{}".format(sgen.pop()))

    s3fgen = Sphere3Hopf(base);
    for _ in range(10):
        print("{}".format(s3fgen.pop()))
