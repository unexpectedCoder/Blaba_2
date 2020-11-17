from typing import List, Tuple, Union
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
        self.particles = None

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
        if self.particles is not None:
            b.particles = self.particles
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
            self.particles += self._pos

    @property
    def particles(self) -> List[Particle]:
        """Массив координат частиц."""
        return self._parts

    @particles.setter
    def particles(self, parts: Union[List[Particle], None]):
        if parts is not None:
            self._parts = parts.copy()
        else:
            self._parts = None

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
        nw, nh = int(self.width / dw), int(self.height / dh)                # кол-во частиц по ширине и высоте

        positions = []
        y1 = lambda x: np.tan(np.deg2rad(15)) * (x - self.width) + .5 * self.height
        y2 = lambda x: -np.tan(np.deg2rad(15)) * (x - self.width) + .5 * self.height
        for i in range(nh):         # вверх по стенке
            for j in range(nw):     # поперёк стенки
                h = i * dh
                w = j * dw if i % 2 == 0 else j * dw + r
                if kind == 'striker':
                    if y1(w) < h < y2(w):
                        positions.append([w, h])
                else:
                    positions.append([w, h])
        positions = np.array(positions, dtype=np.float64)
        positions += self.pos
        if center:
            positions[:, 1] -= .5 * self.height

        dm = self.mass / positions.size     # масса одной частицы

        # Вращение
        a = self.rotate
        if a != 0:
            positions[:, 0] -= self.width
            rotated_pos = np.array([np.matmul([[np.cos(a), np.sin(a)],
                                               [-np.sin(a), np.cos(a)]], p.T) for p in positions])
            rotated_pos[:, 0] += self.width
            self.particles = [Particle(velo=np.array([0., 0.]), pos=pos, m=dm, color=self.color)
                              for pos in rotated_pos]
        else:
            self.particles = [Particle(velo=np.array([0., 0.]), pos=pos, m=dm, color=self.color)
                              for pos in positions]

    def get_draw_particles(self, scale: np.ndarray, win_size: Tuple[int, int]) -> np.ndarray:
        """Преобразование физических координат в экранные координаты."""
        draw_parts = np.array([p.pos.copy() for p in self.particles], order='F')
        draw_parts[:, 1] += (win_size[1] // 2) / scale[1]   # центрирование на экране относительно оси Ox
        return draw_parts * scale

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
