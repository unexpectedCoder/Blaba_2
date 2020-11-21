from tqdm import tqdm

from body import Body
from space import Space
from cell import Cell


class Solver:
    """Класс решателя уравнений динамики частиц."""

    def __init__(self):
        self.cells = None

    @staticmethod
    def check_bodies_particles(w: Body, s: Body):
        if w.particles is None or s.particles is None:
            raise ValueError("тела не разбиты на частицы!")

    def gen_mesh(self, space: Space, rc: float):
        """Сгенерировать (создать) сетку."""
        print("Генерация сетки...")

        side = 2 * rc
        size = side, side
        nc, nr = int(space.w / size[0]), int(space.h / size[1])

        cells = []
        for i in range(nr):
            cells.append([Cell(size=size, pos=(j*size[1], i*size[0] - .5*space.h + size[0]))
                          for j in range(nc)])
        self.cells = cells

        print("Сетка создана!")

    def fill_mesh(self, b: Body):
        print("Заполнение ячеек частицами...")

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
        print("Готово!")

    def clear_mesh(self):
        if self.cells:
            for row in self.cells:
                for c in row:
                    c.clear()
