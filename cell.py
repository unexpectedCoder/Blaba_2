from typing import Iterable, List, Tuple
import numpy as np

from particle import Particle


class Cell:
    """Класс прямоугольной клетки, содержащей частицы."""

    def __init__(self, size: Tuple[float, float], pos: Tuple[float, float]):
        self._size = np.array(size)
        self._dl = np.array(pos)            # верхняя левая вершина
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

    def add_particles(self, parts: List[Particle]):
        """Добавить несколько частиц."""
        self._parts += parts.copy()

    def rm_particles(self, ps: List[Particle]):
        """Удалить несколько частиц по указанным индексам."""
        for p in ps:
            self._parts.remove(p)

    def add_particle(self, p: Particle):
        """Добавить частицу."""
        self._parts.append(p.copy())

    def rm_particle(self, p: Particle):
        """Удалить заданную частицу."""
        self._parts.remove(p)

    def is_empty(self) -> bool:
        """Пуста ли ячейка (*True*) или нет (*False*)."""
        if self.particles:
            return False
        return True


if __name__ == '__main__':
    print(Cell(size=(1, 1), pos=(2, 3)))
