from typing import Iterable, Union
import numpy as np


class SphereParticle:
    """Сферическая частица."""

    def __init__(self, pos: Union[Iterable, np.ndarray], r: float):
        self.pos = pos
        self._r = r

    def __repr__(self):
        return f"{self.__class__.__name__}: r={self.r} pos={self.pos}"

    @property
    def r(self) -> float:
        """Радиус частицы."""
        return self._r

    @property
    def pos(self) -> np.ndarray:
        """Дискретная позиция частицы -- целая или половинчатая."""
        return self._pos

    @pos.setter
    def pos(self, pos: Union[Iterable, np.ndarray]):
        if not isinstance(pos, np.ndarray):
            self._pos = np.array(pos, dtype=np.float64)
        else:
            self._pos = pos.copy()
