from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import pickle

from body import Body
from space import Space
from particle import Particle
from cell import Cell
from mesh import Mesh


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

        self.mesh = None
        self._striker_parts = []
        self._wall_parts = []

    @staticmethod
    def check_bodies_particles(w: Body, s: Body):
        if w.particles is None or s.particles is None:
            raise ValueError("тела не разбиты на частицы!")

    def create_mesh(self, load: bool):
        """Создать сетку.

        :param load: флаг -- считывать ли данные из файла (*True*) или построить заново (*False*).
        """
        if load:
            path = 'data/start_mesh'
            print(f"Чтение сетки из файла '{path}'...")
            with open(path, 'rb') as f:
                self.mesh = Mesh(pickle.load(f))
        else:
            print("Генерация сетки...")
            cells = self._build_mesh()
            cells = self._fill_mesh(cells)
            self.mesh = Mesh(cells)
            self.mesh.save()
        print("Сетка создана!")

    def _build_mesh(self) -> List[List[Cell]]:
        print("\tСоздание ячеек...")

        side = 2 * self.rs
        size = side, side
        nc, nr = int(self.space.w / size[0]), int(self.space.h / size[1])

        cells = []
        for i in range(nr):
            cells.append([Cell(size=size, pos=(j*size[1], i*size[0] - .5*self.space.h + size[0]))
                          for j in range(nc)])

        print("\tГотово!")
        return cells

    def _fill_mesh(self, cells: List[List[Cell]]) -> List[List[Cell]]:
        print("\tЗаполнение ячеек частицами...")

        parts = [p for p in self.striker.particles] + [p for p in self.wall.particles]
        for row in tqdm(cells):
            for cell in row:
                rm_parts = []
                for i, p in enumerate(parts):
                    if p in cell:
                        cell.add_particle(p)
                        rm_parts.append(p)
                for rp in rm_parts:
                    parts.remove(rp)
        if parts:
            raise ValueError("Не все частицы попали в сетку!")

        print("\tГотово!")
        return cells

    def relax(self, t_span: Tuple[float, float], dt: float):
        """Запустить процесс релаксации.

        :param t_span: промежуток времени, с.
        :param dt: шаг по времени, с.
        """
        print("Запущен процесс релаксации...")
        self._calc_euler(dt / 10)  # начальное приближение по Эйлеру
        t = t_span[0] + dt / 10
        while t < t_span[1]:
            print(f" - шаг t={t:.6f}")
            self._calc_verlet(dt)
            t += dt
        print("Процесс релаксации завершён!")

    def _calc_euler(self, dt: float):
        print("Расчёт первого приближения Эйлера для старта метода Верле...")
        for i in tqdm(range(1, len(self.mesh.cells[:-1]))):
            for j in range(1, len(self.mesh.cells[1][:-1])):
                cell = self.mesh.cells[i][j]
                if not cell.is_empty():
                    cells = cell, \
                            self.mesh.cells[i-1][j-1], self.mesh.cells[i-1][j], self.mesh.cells[i-1][j+1], \
                            self.mesh.cells[i][j-1], self.mesh.cells[i][j+1], \
                            self.mesh.cells[i+1][j-1], self.mesh.cells[i+1][j], self.mesh.cells[i+1][j+1]

                    self._forces_for_particles(cells)

                    for p in cell.particles:
                        p.velo += p.force / p.mass * dt
                        p.pos += p.velo * dt
        self._update_mesh()

    def solve(self, t_span: Tuple[float, float], dt: float, v0: np.ndarray, body_name: str):
        """Основная функция, решающая систему ДУ для каждого момента времени.

        :param t_span: интервал времени, с.
        :param dt: шаг по времени, с.
        :param v0: вектор начальной скорости ударника, м/с.
        :param body_name: имя тела, для которого назначается начальная скорость *v0*.
        """
        self._set_v0(v0, body_name)

        print("Запущен процесс моделирования взаимодействия ударника со стенкой...")
        t = t_span[0] + dt
        while t < t_span[1]:
            print(f" - шаг t={t:.6f}")
            self._calc_verlet(dt)
            t += dt
        print("Процесс моделирования взаимодействия завершён!")

    def _set_v0(self, v0: np.ndarray, body_name: str):
        print(f"Установка вектора начальной скорости для каждой частицы тела '{body_name}'...")
        for row in tqdm(self.mesh.cells):
            for cell in row:
                for p in cell.particles:
                    if p.name == body_name:
                        p.velo = v0
        print("Готово!")

    def _calc_verlet(self, dt: float):
        for i in range(1, len(self.mesh.cells[:-1])):
            for j in range(1, len(self.mesh.cells[1][:-1])):
                cell = self.mesh.cells[i][j]
                if not cell.is_empty():
                    cells = cell, \
                            self.mesh.cells[i-1][j-1], self.mesh.cells[i-1][j], self.mesh.cells[i-1][j+1], \
                            self.mesh.cells[i][j-1], self.mesh.cells[i][j+1], \
                            self.mesh.cells[i+1][j-1], self.mesh.cells[i+1][j], self.mesh.cells[i+1][j+1]

                    self._forces_for_particles(cells)

                    for p in cell.particles:
                        buf = p.pos.copy()
                        p.pos = 2*p.pos - p.pos_prev + p.force / p.mass * dt**2
                        # p.velo += .5 * (p.pos - p.pos_prev) / dt
                        # p.pos += p.velo * dt
                        p.pos_prev = buf
        self._update_mesh()

    def _forces_for_particles(self, cells: Tuple[Cell, ...]):
        cell = cells[0]
        neighbor_cells = cells[1:]

        for pi in cell.particles:
            pi.force = np.array([0., 0.])

            # Воздействие от частиц в центральной ячейке
            for pj in cell.particles:
                if pj != pi:
                    pi.force -= self.dU(pi.pos - pj.pos)

            # Воздействие от частиц соседних ячеек
            for neighbor in neighbor_cells:
                if not neighbor.is_empty():
                    for p in neighbor.particles:
                        pi.force -= self.dU(pi.pos - p.pos)

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

    def _update_mesh(self):
        for i in range(1, len(self.mesh.cells) - 1):
            for j in range(1, len(self.mesh.cells[1][:]) - 1):
                cell = self.mesh.cells[i][j]
                outer_parts = [p for p in cell.particles if p not in cell]  # частицы, вылетевшие из ячейки
                cell.rm_particles(outer_parts)
                self._distribute_particles(outer_parts, i, j)

    def _distribute_particles(self, parts: List[Particle], i: int, j: int):
        # Распределить вылетевшие частицы по соседним ячейкам
        for p in parts:
            if p in self.mesh.cells[i-1][j-1]:
                self.mesh.cells[i-1][j-1].add_particle(p)
            elif p in self.mesh.cells[i-1][j]:
                self.mesh.cells[i-1][j].add_particle(p)
            elif p in self.mesh.cells[i-1][j+1]:
                self.mesh.cells[i-1][j+1].add_particle(p)
            elif p in self.mesh.cells[i][j-1]:
                self.mesh.cells[i][j-1].add_particle(p)
            elif p in self.mesh.cells[i][j+1]:
                self.mesh.cells[i][j+1].add_particle(p)
            elif p in self.mesh.cells[i+1][j-1]:
                self.mesh.cells[i+1][j-1].add_particle(p)
            elif p in self.mesh.cells[i+1][j]:
                self.mesh.cells[i+1][j].add_particle(p)
            elif p in self.mesh.cells[i+1][j+1]:
                self.mesh.cells[i+1][j+1].add_particle(p)
