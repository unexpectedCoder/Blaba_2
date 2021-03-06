from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os

from particle import Particle


class Body:
    """Класс некоторого физического тела.

    Свойства:
       * *mass* -- масса тела в кг;
       * *width* -- ширина в м;
       * *height* -- высота в м;
       * *size* -- кортеж размера вида (ширина; высота) в м;
       * *name* -- название тела;
       * *color* -- цвет тела для отрисовки;
       * *pos* -- позиция левого верхнего угла описанного прямоугольника, м;
       * *rotate* -- угол поворота, рад.
    """

    def __init__(self,
                 mass: float,
                 size: Tuple[float, float],
                 name: str,
                 color: Tuple[int, int, int],
                 pos: np.ndarray,
                 rotate_deg: float = 0.):
        self._parts = []

        self.mass = mass
        self.size = size
        self.name = name
        self.color = color
        self.pos = pos
        self.rotate = np.deg2rad(rotate_deg)

    def __repr__(self):
        return f"{self.__class__.__name__}:" \
               f" name='{self.name}'" \
               f" mass={self.mass}" \
               f" (w; h)={self.size}" \
               f" color={self.color}" \
               f" pos={self.pos}" \
               f" rotate={np.rad2deg(self.rotate)}"

    def copy(self) -> 'Body':
        """:return: Копия экземпляра типа ``Body``."""
        b = Body(self.mass, self.size, self.name, self.color, self.pos,
                 rotate_deg=np.rad2deg(self.rotate))
        if self._parts:
            b._parts = [p.copy() for p in self._parts]
        return b

    @property
    def mass(self) -> float:
        """Масса тела, кг."""
        return self._m

    @mass.setter
    def mass(self, m: float):
        self._m = m

    @property
    def width(self) -> float:
        """Ширина, м."""
        return self._w

    @width.setter
    def width(self, w: float):
        self._w = w

    @property
    def height(self) -> float:
        """Высота, м."""
        return self._h

    @height.setter
    def height(self, h: float):
        self._h = h

    @property
    def size(self) -> Tuple[float, float]:
        """Размер тела ``(ширина; высота)``, м."""
        return self._w, self._h

    @size.setter
    def size(self, s: Tuple[float, float]):
        self._w, self._h = s

    @property
    def name(self) -> str:
        """Название тела."""
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n

    @property
    def color(self) -> Tuple[int, int, int]:
        """Цвет тела."""
        return self._color

    @color.setter
    def color(self, c: Tuple[int, int, int]):
        self._color = c

    @property
    def pos(self) -> np.ndarray:
        """Координата тела в пространстве (левый верхний угол описанного прямоугольника), м."""
        return self._pos

    @pos.setter
    def pos(self, p: np.ndarray):
        self._pos = p.copy()
        if self._parts is not None:
            for p in self._parts:
                p.pos += self._pos

    @property
    def particles(self) -> List[Particle]:
        """Массив координат частиц."""
        return self._parts

    @particles.setter
    def particles(self, ps: List[Particle]):
        self._parts = [p.copy() for p in ps]

    @property
    def rotate(self) -> float:
        """Угол поворота."""
        return self._rot

    @rotate.setter
    def rotate(self, rot: float):
        self._rot = rot

    def break_into_particles(self, n: int, dim: str, kind: str, center: bool = True):
        """Разбить тело на частицы.

        :param n: количество частиц по измерению ``dim``.
        :param dim: направление (измерение) по которому тело разбивается на ``n`` частиц -- ``w`` или ``'h'``.
                    Количество частиц в другом направлении зависит от ``n`` и получается автоматически.
        :param center: флаг -- центрировать ли частицы относительно оси симметрии (середины высоты тела).
        :param kind: тип тела -- 'wall' или 'striker' (имеет заострённый носик).
        """
        if dim != 'w' and dim != 'h':
            raise ValueError

        r = .5 * self.width / n if dim == 'w' else .5 * self.height / n     # радиус частицы
        dw, dh = 2 * r, r * np.sqrt(3.)                                     # шаг координат по ширине и высоте
        nw, nh = int(self.width / dw) + 1, int(self.height / dh) + 1                # кол-во частиц по ширине и высоте

        positions = []
        y1 = lambda x: np.tan(np.deg2rad(15)) * (x - self.width) + .5 * self.height
        y2 = lambda x: -np.tan(np.deg2rad(15)) * (x - self.width) + .5 * self.height

        for i in range(nh):         # по высоте стенки
            for j in range(nw):     # по ширине стенки
                y = i * dh
                x = j * dw if i % 2 == 0 else j * dw + r
                if kind == 'striker':
                    if y1(x) < y < y2(x):
                        positions.append([x, y])
                else:
                    positions.append([x, y])
        positions = np.array(positions)
        positions += self.pos

        if center:
            positions[:, 1] -= .5 * self.height

        dm = self.mass / positions.size     # масса одной частицы

        # Вращение
        a = self.rotate
        if a != 0:
            positions[:, 0] -= self.width + self.pos[0]
            rotated_pos = np.array([np.matmul([[np.cos(a), -np.sin(a)],
                                               [np.sin(a), np.cos(a)]], p.T) for p in positions])
            rotated_pos[:, 0] += self.width + self.pos[0]
            self._parts = [Particle(name=self.name, velo=np.array([0., 0.]), pos=pos, m=dm, color=self.color)
                           for pos in rotated_pos]
        else:
            self._parts = [Particle(name=self.name, velo=np.array([0., 0.]), pos=pos, m=dm, color=self.color)
                           for pos in positions]

    def save_image(self):
        plt.figure("Body", figsize=(6, 6))
        parts = np.array([p.pos.copy() for p in self.particles], order='F')
        x, y = parts[:, 0], parts[:, 1]
        plt.scatter(x, y, color='k', marker='.')
        plt.title(self.name)
        plt.xlabel("ширина, м")
        plt.ylabel("высота, м")

        if not os.path.isdir('pics'):
            os.mkdir('pics')
        plt.savefig(f"pics/{self.name}.png")


if __name__ == '__main__':
    print(Body(1, (100, 300), 'somebody', (255, 0, 0), np.array([20, 0])))
