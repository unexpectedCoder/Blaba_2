from body import Body
from space import Space
from cell import Cell

import numpy as np


class Solver:
    """Класс решателя уравнений динамики частиц."""

    def __init__(self, wall: Body, striker: Body, space: Space,
                 sigma: float, epsilon: float):
        self.check_bodies_particles(wall, striker)

        self.wall = wall
        self.striker = striker
        self.space = space
        self.sigma = sigma
        self.epsilon = epsilon

        self.rs = (26/7)**(1/6) * self.sigma
        self.rc = 67/48 * self.rs
        self.k1 = -387072/61009 * self.epsilon / self.rs**3
        self.k2 = -24192/3211 * self.epsilon / self.rs**2

        self.cells = None

    @staticmethod
    def check_bodies_particles(w: Body, s: Body):
        if w.particles is None or s.particles is None:
            raise ValueError("тела не разбиты на частицы!")

    def dU(self, dr: np.ndarray) -> np.ndarray:
        """Градиент потенциала Леннарда-Джонса.

        :param dr: вектор расстояния между двумя частицами.
        :return: Градиент потенциала Леннарда-Джонса со сглаживанием сплайнами.
        """
        d = np.sqrt(dr[0]**2 + dr[1]**2)
        if d <= self.rs:
            return dr / d * \
                   24 * self.epsilon * self.sigma**6 * \
                   (d**6 - 2 * self.sigma**6) / d**13
        if self.rs < d <= self.rc:
            return dr / d * \
                   (d - self.rc) * \
                   (3*self.k1*(d - self.rc) + 2*self.k2)
        return np.array([0., 0.])

    def build_mesh(self):
        """Построить сетку."""
        side = self.rs
        size = side, side
        nc, nr = int(self.space.w / size[0]), int(self.space.h / size[1])
        self.cells = [Cell(size=size, pos=(j*size[1], i*size[0])) for i in range(nr) for j in range(nc)]
