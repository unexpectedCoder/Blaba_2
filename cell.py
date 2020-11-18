from typing import Iterable, List, Tuple
import numpy as np

from particle import Particle


class Cell:
    """Класс прямоугольной клетки, содержащей частицы."""

    def __init__(self, size: Tuple[float, float], pos: Tuple[float, float]):
        self._size = np.array(size)
        self._ul = np.array(pos)            # верхняя левая вершина
        self._dr = self._ul + self._size    # нижняя правая вершина

        self._parts = []                    # список частиц

    def __repr__(self):
        return f"{self.__class__.__name__}:" \
               f" size={self.size}" \
               f" ul(pos)={self.ul}" \
               f" dr={self.dr}" \
               f" n_particles={len(self.particles)}"

    def __contains__(self, p: Particle) -> bool:
        return self.ul[0] <= p.pos[0] <= self.dr[0] and self.ul[1] <= p.pos[1] <= self.dr[1]

    @property
    def size(self) -> np.ndarray:
        return self._size

    @property
    def ul(self) -> np.ndarray:
        """Upper-left вершина."""
        return self._ul

    @property
    def dr(self) -> np.ndarray:
        """Down-right вершина."""
        return self._dr

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
