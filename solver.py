from body import Body
from space import Space
from cell import Cell
from mesh import Mesh


class Solver:
    """Класс решателя уравнений динамики частиц."""

    def __init__(self, wall: Body, striker: Body, space: Space,
                 sigma: float, epsilon: float):
        self._wall = wall
        self._striker = striker
        self._space = space
        self._sig = sigma
        self._eps = epsilon

    @property
    def sigma(self) -> float:
        return self._sig

    @property
    def epsilon(self) -> float:
        return self._eps

    def build_mesh(self) -> Mesh:
        """Построить сетку из ячеек, заполненных частицами."""
        pass    # TODO
