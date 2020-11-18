from typing import Tuple
from uuid import uuid4
import numpy as np


class Particle:
    """Класс частицы тела."""

    def __init__(self, velo: np.ndarray, pos: np.ndarray, m: float, color: Tuple[int, int, int],
                 uuid: int = None):
        self.force = np.array([0., 0.])
        self.velo = velo
        self.pos_prev, self.pos = pos, pos
        self.mass = m
        self.color = color

        self.uuid = uuid4().int if uuid is None else uuid

    def __repr__(self):
        return f"{self.__class__.__name__}:" \
               f" velo={self.velo}" \
               f" pos={self.pos}" \
               f" mass={self.mass}" \
               f" color={self.color}"

    def __hash__(self):
        return self.uuid

    def __eq__(self, p: 'Particle'):
        return p.uuid == self.uuid

    def copy(self) -> 'Particle':
        """:return: Копия экземпляра."""
        p = Particle(self.velo, self.pos, self.mass, self.color,
                     uuid=self.uuid)
        p.pos_prev = self.pos_prev
        return p

    @property
    def force(self) -> np.ndarray:
        return self._f

    @force.setter
    def force(self, f: np.ndarray):
        self._f = f

    @property
    def velo(self) -> np.ndarray:
        """Вектор скорости, м/с."""
        return self._v

    @velo.setter
    def velo(self, v: np.ndarray):
        self._v = v

    @property
    def pos(self) -> np.ndarray:
        """Радиус вектор частицы (координаты), м."""
        return self._pos

    @pos.setter
    def pos(self, p: np.ndarray):
        self._pos = p

    @property
    def pos_prev(self) -> np.ndarray:
        """Координата частицы в предшествующий момент времени."""
        return self._pos_prev

    @pos_prev.setter
    def pos_prev(self, pp: np.ndarray):
        self._pos_prev = pp

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

    def reset_force(self):
        """Сбросить суммарный вектор силы в ноль."""
        self.force = np.array([0., 0.])


if __name__ == '__main__':
    print(Particle(velo=np.array([10, 12.5]), pos=np.array([0., -1.75]), m=.05, color=(182, 75, 200)))
