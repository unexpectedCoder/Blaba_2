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
            self._build_mesh()
        print("Сетка создана!")

    def _build_mesh(self):
        print("Генерация сетки...")

        side = self.rs
        size = side, side
        nc, nr = int(self.space.w / size[0]), int(self.space.h / size[1])

        print("Создание ячеек...")
        cells = []
        for i in range(nr):
            cells.append([Cell(size=size, pos=(j * size[1], i * size[0] - .5 * self.space.h)) for j in range(nc)])
        parts = [p for p in self.wall.particles] + [p for p in self.striker.particles]

        print("Заполнение ячеек частицами...")
        for row in tqdm(cells):
            for cell in row:
                for i, p in enumerate(parts):
                    if p in cell:
                        cell.add_particle(parts.pop(i))

        self.mesh = Mesh(cells)
        self.mesh.save()

    def relax(self, t_span: np.ndarray, dt: float):
        """Запустить процесс релаксации.

        :param t_span: промежуток времени, с.
        :param dt: шаг по времени, с.
        """
        print("Запущен процесс релаксации...")

        self._calc_euler(dt)    # начальное приближение

        print("Процесс релаксации завершён!")

    def _calc_euler(self, dt: float):
        print("Расчёт первого приближения Эйлера для старта метода Верле...")
        for i in tqdm(range(1, len(self.mesh.cells) - 1)):
            for j in range(1, len(self.mesh.cells[1][:]) - 1):
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
        self._reset()
        self._update_mesh()

    def _forces_for_particles(self, cells: Tuple[Cell, ...]):
        cell = cells[0]
        neighbor_cells = cells[1:]

        for i in range(len(cell.particles) - 1):
            # Воздействие от частиц в центральной ячейке
            for j in range(i+1, len(cell.particles)):
                dr = cell.particles[i].pos - cell.particles[j].pos
                f = -self.dU(dr)
                cell.particles[i].force += f
                cell.particles[j].force -= f                        # 3-й закон Ньютона

            # Воздействие от частиц соседних ячеек
            for neighbor in neighbor_cells:
                if not neighbor.is_empty():
                    for p in neighbor.particles:
                        dr = cell.particles[i].pos - p.pos
                        cell.particles[i].force -= self.dU(dr)

    def _reset(self):
        for row in self.mesh.cells:
            for cell in row:
                for p in cell.particles:
                    p.reset_force()

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
            else:
                self.mesh.cells[i+1][j+1].add_particle(p)

    # TODO начать решать!!!
    # Функция solve должна обходить все частицы, решая для них систему ОДУ и
    # сохраняя результаты в файл numpy (*.npz, скорее всего)
    def solve(self, dt: float):
        """Основная функция, решающая систему ДУ для каждого момента времени.

        :param dt: шаг по времени, с.
        """
        pass
