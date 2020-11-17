from typing import Tuple
import numpy as np


class Particle:
    """Класс частицы тела."""

    def __init__(self, velo: np.ndarray, pos: np.ndarray, m: float, color: Tuple[int, int, int]):
        self.velo = velo
        self.pos = pos
        self.mass = m
        self.color = color

    def __repr__(self):
        return f"{self.__class__.__name__}:" \
               f" velo={self.velo}" \
               f" pos={self.pos}" \
               f" mass={self.mass}" \
               f" color={self.color}"

    def copy(self) -> 'Particle':
        """:return: Копия экземпляра."""
        return Particle(self.velo, self.pos, self.mass, self.color)

    @property
    def velo(self) -> np.ndarray:
        """Вектор скорости, м/с."""
        return self._v

    @velo.setter
    def velo(self, v: np.ndarray):
        self._v = v.copy()

    @property
    def pos(self) -> np.ndarray:
        """Радиус вектор частицы (координаты), м."""
        return self._pos

    @pos.setter
    def pos(self, p: np.ndarray):
        self._pos = p.copy()

    @property
    def mass(self) -> float:
        """Масса частицы, кг."""
        return self._m

    @mass.setter
    def mass(self, m: float):
        self._m = m

    @property
    def color(self) -> Tuple[int, int, int]:
        """RGB цвет частицы для отрисовки."""
        return self._color

    @color.setter
    def color(self, c: Tuple[int, int, int]):
        self._color = c


if __name__ == '__main__':
    print(Particle(velo=np.array([10, 12.5]), pos=np.array([0., -1.75]), m=.05, color=(182, 75, 200)))
