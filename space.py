from typing import Tuple


class Space:
    """Класс, физическое описывающий пространство.

    Свойства:
       * *size* -- размер (ширина, высота), м;
       * *w* -- ширина, м;
       * *h* -- высота, м.
    """

    def __init__(self, size: Tuple[float, float]):
        self._size = size

    def __repr__(self):
        return f"{self.__class__.__name__}: size={self.size}"

    def copy(self) -> 'Space':
        return Space(self.size)

    @property
    def size(self) -> Tuple[float, float]:
        return self._size

    @property
    def w(self) -> float:
        return self._size[0]

    @property
    def h(self) -> float:
        return self._size[1]
