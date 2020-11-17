from typing import List

from cell import Cell


class Mesh:
    """Класс сетки, содержащей клетки типа ``Cell``."""

    def __init__(self, cells: List[List[Cell]]):
        self._cells = cells

    @property
    def cells(self) -> List[List[Cell]]:
        """Матрица клеток сетки."""
        return self._cells
