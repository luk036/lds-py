import numpy as np
from pytest import approx
from scipy.spatial import ConvexHull

from lds_py.discrep_2 import discrep_2
from lds_py.lds_n import CylinN, SphereN


def run_lds(spgen):
    npoints = 600
    Triples = np.array([spgen.pop() for _ in range(npoints)])
    hull = ConvexHull(Triples)
    triangles = hull.simplices
    return discrep_2(triangles, Triples)


def test_sphere_n():
    spgen = SphereN(3, [2, 3, 5, 7])
    measure = run_lds(spgen)
    assert measure == approx(0.9125914)
    # assert measure < 0.913
    # assert measure > 0.912


def test_cylin_n():
    cygen = CylinN(3, [2, 3, 5, 7])
    measure = run_lds(cygen)
    assert measure == approx(1.0505837105828988)
    # assert measure < 1.086
    # assert measure > 1.085
