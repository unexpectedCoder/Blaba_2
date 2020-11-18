from typing import List
import os
import pickle

from cell import Cell


class Mesh:
    """Класс сетки, содержащей клетки типа ``Cell``."""

    def __init__(self, cells: List[List[Cell]]):
        self._cells = cells

    @property
    def cells(self) -> List[List[Cell]]:
        """Матрица клеток сетки."""
        return self._cells

    def save_cells(self):
        """Сохранить сетку (матрицу с частицами) с помощью модуля *pickle*."""
        if not os.path.isdir('data'):
            os.mkdir('data')
        path = 'data/start_mesh'
        with open(path, 'wb') as f:
            pickle.dump(self.cells, f)
