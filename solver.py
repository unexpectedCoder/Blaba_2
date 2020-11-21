from tqdm import tqdm
import numpy as np

from body import Body
from space import Space
from cell import Cell


class Solver:
    """Класс решателя уравнений динамики частиц."""

    def __init__(self, sigma: float, epsilon: float):
        self.sigma = sigma
        self.epsilon = epsilon

        self.rs = (26 / 7) ** (1 / 6) * self.sigma
        self.rc = 67 / 48 * self.rs
        self.k1 = -387072 / 61009 * self.epsilon / self.rs ** 3
        self.k2 = -24192 / 3211 * self.epsilon / self.rs ** 2

        self.cells = None

    def dU(self, dr: np.ndarray) -> np.ndarray:
        """Градиент потенциала Леннарда-Джонса."""
        d = np.sqrt(dr[0] ** 2 + dr[1] ** 2)
        if d <= self.rs:
            return dr / d * \
                   24 * self.epsilon * self.sigma ** 6 * \
                   (d ** 6 - 2 * self.sigma ** 6) / d ** 13
        if self.rs < d <= self.rc:
            return dr / d * \
                   (d - self.rc) * \
                   (3 * self.k1 * (d - self.rc) + 2 * self.k2)
        return np.array([0., 0.])

    @staticmethod
    def check_bodies_particles(w: Body, s: Body):
        if w.particles is None or s.particles is None:
            raise ValueError("тела не разбиты на частицы!")

    def gen_mesh(self, space: Space):
        """Сгенерировать (создать) сетку."""
        print("Генерация сетки...")

        side = 2 * self.rc
        size = side, side
        nc, nr = int(space.w / size[0]), int(space.h / size[1])

        cells = []
        for i in range(nr):
            cells.append([Cell(size=size, pos=(j*size[1], i*size[0] - .5*space.h + size[0]))
                          for j in range(nc)])
        self.cells = cells

        print("Сетка создана!")

    def relax(self, b: Body) -> Body:
        b = b.copy()
        self._fill_mesh(b)
        # TODO Расчёт
        return b

    def _fill_mesh(self, b: Body):
        print(f"\tЗаполнение ячеек частицами '{b.name}'...")

        parts = b.particles.copy()
        for row in tqdm(self.cells):
            for c in row:
                rm_parts = []
                for p in parts:
                    if p in c:
                        c.add_particle(p)
                        rm_parts.append(p)
                # Удаление вошедших в текущую ячейку частиц
                for rp in rm_parts:
                    parts.remove(rp)

        if parts:
            raise ValueError("Не все частицы попали в сетку!")
        print("\tГотово!")

    def clear_mesh(self):
        if self.cells:
            for row in self.cells:
                for c in row:
                    c.clear()

    def solve(self, wall: Body, striker: Body):
        wall = wall.copy()
        striker = striker.copy()
        self._fill_mesh(wall)
        self._fill_mesh(striker)
        # TODO Расчёт
