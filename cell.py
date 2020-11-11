from typing import Iterable, Tuple, Union
import numpy as np


class Cell:
    """Класс прямоугольной клетки, содержащей частицы."""

    def __init__(self, size: Tuple[float, float], pos: Tuple[float, float]):
        self._size = np.array(size)
        self._ul = np.array(pos)            # верхняя левая вершина
        self._dr = self._ul + self._size    # нижняя правая вершина

        self._parts = None                  # список частиц

    @property
    def size(self) -> np.ndarray:
        return self._size

    @property
    def ul(self) -> np.ndarray:
        """Upper left вершина."""
        return self._ul

    @property
    def dr(self) -> np.ndarray:
        """Down right вершина."""
        return self._dr

    @property
    def particles(self) -> Union[np.ndarray, None]:
        """Массив принадлежащих частиц."""
        return self._parts

    def set_particles(self, parts: np.ndarray):
        """Заполнить частицами."""
        self._parts = parts.copy()

    def delete_particles(self, indexes: Iterable):
        """Удаление частиц по заданным индексам."""
        self._parts = np.delete(self._parts, np.s_(indexes))
