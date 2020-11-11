from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import os


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

        self._dm = None     # масса одной частицы

    def __repr__(self):
        return f"{self.__class__.__name__}:" \
               f" name='{self.name}'" \
               f" mass={self.mass}" \
               f" (w; h)={self.size}" \
               f" color={self.color}" \
               f" pos={self.pos}" \
               f" rotate={np.rad2deg(self.rotate)}" \
               f" dm = {self.dm}"

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
        return self._mass

    @mass.setter
    def mass(self, m: float):
        self._mass = m

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
    def particles(self) -> np.ndarray:
        """Массив координат частиц."""
        return self._parts

    @particles.setter
    def particles(self, parts: Union[np.ndarray, None]):
        if parts is not None:
            self._parts = parts.copy()
        else:
            self._parts = None

    @property
    def rotate(self) -> float:
        """Угол поворота."""
        return self._rotate

    @rotate.setter
    def rotate(self, rot: float):
        self._rotate = rot

    @property
    def dm(self) -> float:
        """Масса частицы."""
        return self._dm

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

        parts = []
        y1 = lambda x: np.tan(np.deg2rad(15)) * (x - self.width) + .5 * self.height
        y2 = lambda x: -np.tan(np.deg2rad(15)) * (x - self.width) + .5 * self.height
        for i in range(nh):         # вверх по стенке
            for j in range(nw):     # поперёк стенки
                h = i * dh
                w = j * dw if i % 2 == 0 else j * dw + r
                if kind == 'striker':
                    if y1(w) < h < y2(w):
                        parts.append([w, h])
                else:
                    parts.append([w, h])
        parts = np.array(parts, dtype=np.float64)
        parts += self.pos
        if center:
            parts[:, 1] -= .5 * self.height

        a = self.rotate
        # Вращение
        if self.rotate != 0:
            parts[:, 0] -= self.width
            rotated = [np.matmul([[np.cos(a), np.sin(a)],
                                  [-np.sin(a), np.cos(a)]], p.T) for p in parts]
            self.particles = np.array(rotated, dtype=np.float64)
            self.particles[:, 0] += self.width
        else:
            self.particles = np.array(parts)

        self._dm = self.mass / self._parts.size  # масса одной частицы

    def get_draw_particles(self, scale: np.ndarray, win_size: Tuple[int, int]) -> np.ndarray:
        """Преобразование физических координат в экранные координаты."""
        draw_parts = self.particles.copy()
        draw_parts[:, 1] += (win_size[1] // 2) / scale[1]   # центрирование на экране по оси Ox
        return draw_parts * scale

    def save_image(self):
        plt.figure("Body", figsize=(6, 6))
        x, y = self.particles[:, 0], self.particles[:, 1]
        plt.scatter(x, y, color='k', marker='.')
        plt.title(self.name)
        plt.xlabel("ширина, м")
        plt.ylabel("высота, м")

        if not os.path.isdir('pics'):
            os.mkdir('pics')
        plt.savefig(f"pics/{self.name}.png")


if __name__ == '__main__':
    print(Body(1, (100, 300), 'somebody', (255, 0, 0), np.array([20, 0])))
