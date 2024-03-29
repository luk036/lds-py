from typing import Sequence, List


def vdc_i(k: int, base: int = 2, scale: int = 10) -> int:
    """
    The function `vdc_i` converts a given number `k` from base `base` to a decimal number using a
    specified scale.

    :param k: The parameter `k` represents the number for which we want to calculate the van der Corput
    sequence value
    :type k: int
    :param base: The `base` parameter represents the base of the number system being used. In this case,
    it is set to 2, which means the number system is binary (base 2), defaults to 2
    :type base: int (optional)
    :param scale: The `scale` parameter determines the precision or number of digits after the decimal
    point in the resulting VDC (Van der Corput) sequence. It specifies the number of times the base is
    raised to calculate the factor, defaults to 10
    :type scale: int (optional)
    :return: The function `vdc_i` returns an integer value.

    Examples:
        >>> vdc_i(1, 2, 10)
        512
    """
    vdc: int = 0
    factor: int = base**scale
    while k != 0:
        factor //= base
        remainder: int = k % base
        k //= base
        vdc += remainder * factor
    return vdc


# The `VdCorput` class initializes an object with a base and scale value, and sets the count to 0.
class VdCorput:
    def __init__(self, base: int = 2, scale: int = 10) -> None:
        """
        The function initializes an object with a base and scale value, and sets the count to 0.

        :param base: The `base` parameter is an optional integer argument that specifies the base of the
        number system. By default, it is set to 2, which means the number system is binary (base 2).
        However, you can change the value of `base` to any other prime number to use a different, defaults to 2
        :type base: int (optional)
        :param scale: The `scale` parameter determines the number of digits that can be represented in the
        number system. For example, if `scale` is set to 10, the number system can represent digits from 0
        to 9, defaults to 10
        :type scale: int (optional)
        """
        self._base: int = base
        self._scale: int = scale
        self._count: int = 0

    def pop(self) -> int:
        """
        The `pop()` function is a member function of the `VdCorput` class that increments the count and
        calculates the next value in the Van der Corput sequence.
        :return: The `pop()` function is returning an `int` value.

        Examples:
            >>> vdc = VdCorput(2, 10)
            >>> vdc.pop()
            512
        """
        self._count += 1
        return vdc_i(self._count, self._base, self._scale)

    def reseed(self, seed: int) -> None:
        """reseed

        The `reseed(size_t seed)` function is used to reset the state of the
        sequence generator to a specific seed value. This allows the sequence
        generator to start generating the sequence from the beginning, or from
        a specific point in the sequence, depending on the value of the seed.

        Args:
            seed (int): _description_

        Examples:
            >>> vdc = VdCorput(2, 10)
            >>> vdc.reseed(0)
            >>> vdc.pop()
            512
        """
        self._count = seed


class Halton:
    """Halton sequence generator

    The `Halton` class is a sequence generator that generates points in a
    2-dimensional space using the Halton sequence. The Halton sequence is a
    low-discrepancy sequence that is often used in quasi-Monte Carlo methods.
    It is generated by iterating over two different bases and calculating the
    fractional parts of the numbers in those bases. The `Halton` class keeps
    track of the current count and bases, and provides a `pop()` method that
    returns the next point in the sequence as a `List[int]`.

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
        """
        The `__init__()` function is a constructor for the `Halton` class that initializes two `VdCorput`
        objects with the given bases.

        :param base: The `base` parameter is a list of two integers. These integers are used as the bases
        for generating the Halton sequence. The first integer in the list is used as the base for generating
        the first component of the sequence, and the second integer is used as the base for generating the
        second component
        :type base: Sequence[int]
        """
        self._vdc0 = VdCorput(base[0], scale[0])
        self._vdc1 = VdCorput(base[1], scale[1])

    def pop(self) -> List[int]:
        """
        The `pop` function returns a list of two integers by popping elements from `vdc0` and `vdc1`.
        :return: The `pop` method is returning a list of two integers.
        """
        return [self._vdc0.pop(), self._vdc1.pop()]

    def reseed(self, seed: int) -> None:
        """reseed

        The `reseed(size_t seed)` function is used to reset the state of the
        sequence generator to a specific seed value. This allows the sequence
        generator to start generating the sequence from the beginning, or from
        a specific point in the sequence, depending on the value of the seed.

        Args:
            seed (int): _description_
        """
        self._vdc0.reseed(seed)
        self._vdc1.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
