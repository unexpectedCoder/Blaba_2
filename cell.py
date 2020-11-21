from typing import List
import numpy as np

from particle import Particle


class Cell:
    """Класс прямоугольной клетки, содержащей частицы."""

    def __init__(self, size: np.ndarray, pos: np.ndarray):
        self._size = size.copy()
        self._dl = pos.copy()               # верхняя левая вершина
        self._ur = self.dl + self.size      # нижняя правая вершина

        self._parts = []                    # список частиц

    def __repr__(self):
        return f"{self.__class__.__name__}:" \
               f" size={self.size}" \
               f" dl(pos)={self.dl}" \
               f" ur={self.ur}" \
               f" n_particles={len(self.particles)}"

    def __contains__(self, p: Particle) -> bool:
        return self.dl[0] <= p.pos[0] < self.ur[0] and self.dl[1] <= p.pos[1] < self.ur[1]

    def copy(self) -> 'Cell':
        c = Cell(self.size, self.dl)
        c.add_particles(self.particles)
        return c

    @property
    def size(self) -> np.ndarray:
        return self._size

    @property
    def dl(self) -> np.ndarray:
        return self._dl

    @property
    def ur(self) -> np.ndarray:
        return self._ur

    @property
    def particles(self) -> List[Particle]:
        """Массив принадлежащих частиц."""
        return self._parts

    def clear(self):
        self._parts = []

    def add_particles(self, ps: List[Particle]):
        """Добавить несколько частиц."""
        for p in ps:
            self._parts.append(p)

    def rm_particles(self, ps: List[Particle]):
        """Удалить несколько частиц по указанным индексам."""
        for p in ps:
            self._parts.remove(p)

    def add_particle(self, p: Particle):
        """Добавить частицу."""
        self._parts.append(p)

    def rm_particle(self, p: Particle):
        """Удалить заданную частицу."""
        self._parts.remove(p)

    def is_empty(self) -> bool:
        """Пуста ли ячейка (*True*) или нет (*False*)."""
        if self.particles:
            return False
        return True
