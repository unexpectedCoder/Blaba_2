from tqdm import tqdm
from typing import List, Tuple
import numpy as np

from particle import Particle
from body import Body
from space import Space
from cell import Cell


class Solver:
    """Класс решателя уравнений динамики частиц."""

    def __init__(self, wall: Body, striker: Body, space: Space,
                 sigma: float, epsilon: float):
        self.wall = wall.copy()
        self.striker = striker.copy()
        self.space = space.copy()
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

    def gen_mesh(self):
        """Сгенерировать (создать) сетку."""
        print("Генерация сетки...")

        side = 2 * self.rc
        size = np.array([side, side])
        nc, nr = int(self.space.w / size[0]), int(self.space.h / size[1])

        cells = []
        for i in range(nr):
            cells.append([Cell(size=size, pos=np.array([j*size[1], i*size[0] - .5*self.space.h + size[0]]))
                          for j in range(nc)])
        self.cells = cells

        print("Сетка создана!")

    def relax_wall(self, dt: float, t_span: Tuple[float, float]) -> Body:
        self._fill_mesh(self.wall.particles)
        self._calc_euler(dt)
        t = t_span[0] + dt
        while t < t_span[1] + dt:
            print(f"\t t = {t}")
            self._calc_verlet(dt)
            t += dt

        parts = []
        for row in self.cells:
            for c in row:
                for p in c.particles:
                    parts.append(p)
        self.wall.particles = parts

        return self.wall.copy()

    def relax_striker(self, dt: float, t_span: Tuple[float, float]) -> Body:
        self._fill_mesh(self.striker.particles)
        self._calc_euler(dt)
        t = t_span[0] + dt
        while t < t_span[1] + dt:
            print(f"\t t = {t}")
            self._calc_verlet(dt)
            t += dt

        parts = []
        for row in self.cells:
            for c in row:
                for p in c.particles:
                    parts.append(p)
        self.striker.particles = parts

        return self.striker.copy()

    def _fill_mesh(self, parts):
        print(f"\tЗаполнение ячеек частицами '{parts[0].name}'...")

        parts = parts.copy()
        for row in tqdm(self.cells):
            for c in row:
                rm_parts = []
                for p in parts:
                    if p in c:
                        c.add_particle(p.copy())
                        rm_parts.append(p.copy())
                # Удаление вошедших в текущую ячейку частиц
                for rp in rm_parts:
                    parts.remove(rp)

        if parts:
            raise ValueError("Не все частицы попали в сетку!")
        print("\tГотово!")

    def _calc_euler(self, dt: float):
        print("Расчёт первого приближения Эйлера для старта метода Верле...")

        tmp_cells = []
        for row in self.cells:
            r = []
            for cell in row:
                r.append(cell.copy())
            tmp_cells.append(r)

        for i in range(1, len(self.cells[:-1])):
            for j in range(1, len(self.cells[0][:-1])):
                cell = self.cells[i][j]
                if not cell.is_empty():
                    self._calc_force(tmp_cells, cell, i, j)

                    for p in cell.particles:
                        p.velo += p.force * dt
                        p.pos += p.velo * dt

        self._update_mesh()

    def _calc_verlet(self, dt: float):
        tmp_cells = []
        for row in self.cells:
            r = []
            for cell in row:
                r.append(cell.copy())
            tmp_cells.append(r)

        for i in range(1, len(self.cells[:-1])):
            for j in range(1, len(self.cells[0][:-1])):
                cell = self.cells[i][j]
                if not cell.is_empty():
                    self._calc_force(tmp_cells, cell, i, j)

                    for p in cell.particles:
                        buf = p.pos.copy()
                        p.pos = 2*p.pos - p.pos_prev + p.force * dt**2
                        p.velo = .5 * (p.pos - p.pos_prev) / dt

                        p.pos_prev = buf

        self._update_mesh()

    def _calc_force(self, tmp_cells, cell: Cell, i: int, j: int):
        neighbor_cells = tmp_cells[i - 1][j - 1], tmp_cells[i - 1][j], tmp_cells[i - 1][j + 1],\
                         tmp_cells[i][j - 1], tmp_cells[i][j + 1],\
                         tmp_cells[i + 1][j - 1], tmp_cells[i + 1][j], tmp_cells[i + 1][j + 1]

        for pi in cell.particles:
            # Воздействие от частиц в центральной ячейке
            pi.force = np.array([0., 0.])
            for pj in cell.particles:
                if pi != pj:
                    dr = pi.pos - pj.pos
                    pi.force -= self.dU(dr)
            # Воздействие от частиц соседних ячеек
            for neighbor in neighbor_cells:
                if not neighbor.is_empty():
                    for p in neighbor.particles:
                        dr = pi.pos - p.pos
                        pi.force -= self.dU(dr)

    def _update_mesh(self):
        for i in range(1, len(self.cells[:-1])):
            for j in range(1, len(self.cells[1][:-1])):
                cell = self.cells[i][j]
                outer_parts = [p.copy() for p in cell.particles if p not in cell]  # частицы, вылетевшие из ячейки
                self._distribute_particles(outer_parts, i, j)
                cell.rm_particles(outer_parts)

    def _distribute_particles(self, parts: List[Particle], i: int, j: int):
        # Распределить вылетевшие частицы по соседним ячейкам
        for p in parts:
            if p in self.cells[i-1][j-1]:
                self.cells[i-1][j-1].add_particle(p)
            elif p in self.cells[i-1][j]:
                self.cells[i-1][j].add_particle(p)
            elif p in self.cells[i-1][j+1]:
                self.cells[i-1][j+1].add_particle(p)
            elif p in self.cells[i][j-1]:
                self.cells[i][j-1].add_particle(p)
            elif p in self.cells[i][j+1]:
                self.cells[i][j+1].add_particle(p)
            elif p in self.cells[i+1][j-1]:
                self.cells[i+1][j-1].add_particle(p)
            elif p in self.cells[i+1][j]:
                self.cells[i+1][j].add_particle(p)
            elif p in self.cells[i+1][j+1]:
                self.cells[i+1][j+1].add_particle(p)

    def solve(self, wall: Body, striker: Body,
              dt: float, t_span: Tuple[float, float], v0: np.ndarray):
        self._set_v0(striker, v0)

        parts = wall.particles + striker.particles
        self._fill_mesh(parts)

        self._calc_euler(dt)
        t = t_span[0] + dt
        while t < t_span[1] + dt:
            print(f"\t t = {t}")
            self._calc_verlet(dt)
            t += dt

    def _set_v0(self, b: Body, v0: np.ndarray):
        for p in b.particles:
            p.velo += v0

    def clear_mesh(self):
        if self.cells:
            for row in self.cells:
                for c in row:
                    c.clear()
