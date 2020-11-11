from typing import List

from cell import Cell


class Mesh:
    """Класс сетки, содержащей клетки ``Cell``."""
    def __init__(self, cells: List[Cell]):
        self._cells = cells

    @property
    def cells(self) -> List[Cell]:
        """Клетки сетки."""
        return self._cells
