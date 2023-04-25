from typing import Sequence, List


def vdc_i(k: int, base: int = 2, scale: int = 10) -> int:
    """[summary]

    Arguments:
        k (int): number

    Keyword Arguments:
        base (int): [description] (default: {2})

    Returns:
        int: [description]
    """
    vdc: int = 0
    factor: int = base**scale
    while k != 0:
        factor //= base
        remainder: int = k % base
        k //= base
        vdc += remainder * factor
    return vdc


class Vdcorput:
    def __init__(self, base: int = 2, scale: int = 10) -> None:
        """[summary]

        Args:
            base (int, optional): [description]. Defaults to 2.
        """
        self._base: int = base
        self._scale: int = scale
        self._count: int = 0

    def pop(self) -> int:
        """[summary]

        Returns:
            float: [description]
        """
        self._count += 1
        return vdc_i(self._count, self._base, self._scale)

    def reseed(self, seed: int) -> None:
        self._count = seed


class Halton:
    """Generate Halton sequence

    Examples:
        >>> hgen = Halton([2, 3], [11, 7])
        >>> hgen.reseed(0)
        >>> for _ in range(10):
        ...     print(hgen.pop())
        ...
        [1024, 729]
        [512, 1458]
        [1536, 243]
        [256, 972]
        [1280, 1701]
        [768, 486]
        [1792, 1215]
        [128, 1944]
        [1152, 81]
        [640, 810]
    """

    def __init__(self, base: Sequence[int], scale: Sequence[int]) -> None:
        """[summary]

        Args:
            base (Sequence[int]): [description]
            scale (Sequence[int]): [description]
        """
        self._vdc0 = Vdcorput(base[0], scale[0])
        self._vdc1 = Vdcorput(base[1], scale[1])

    def pop(self) -> List[int]:
        """Get the next item

        Returns:
            List[int]:  the next item
        """
        return [self._vdc0.pop(), self._vdc1.pop()]

    def reseed(self, seed: int) -> None:
        """[summary]

        Args:
            seed (int): [description]
        """
        self._vdc0.reseed(seed)
        self._vdc1.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
