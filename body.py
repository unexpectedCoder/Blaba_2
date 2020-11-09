from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


class Body:
    """Класс некоторого физического тела.

    Свойства:
       * *mass* -- масса тела в кг;
       * *width* -- ширина в м;
       * *height* -- высота в м;
       * *size* -- кортеж размера вида (ширина; высота) в м;
       * *name* -- название тела;
       * *color* -- цвет тела для отрисовки;
       * *pos* -- позиция левого верхнего угла описанного прямоугольника, м.
    """

    def __init__(self,
                 mass: float,
                 size: Tuple[float, float],
                 name: str,
                 color: Tuple[int, int, int],
                 pos: np.ndarray):
        self.particles = None

        self.mass = mass
        self.size = size
        self.name = name
        self.color = color
        self.pos = pos

    def __repr__(self):
        return f"{self.__class__.__name__}: " \
               f"name='{self.name}' mass={self.mass} (w; h)={self.size} color={self.color} pos={self.pos}"

    def copy(self) -> 'Body':
        """:return: Копия экземпляра типа ``Body``."""
        b = Body(self.mass, self.size, self.name, self.color, self.pos)
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

    def break_into_particles(self, n: int, dim: str, center: bool = True):
        """Разбить тело на частицы.

        :param n: количество частиц по измерению ``dim``.
        :param dim: направление (измерение) по которому тело разбивается на ``n`` частиц -- ``w`` или ``'h'``.
                    Количество частиц в другом направлении зависит от ``n`` и получается автоматически.
        :param center: фла -- центрировать ли частицы относительно оси симметрии (середины высоты тела).
        """
        if dim != 'w' and dim != 'h':
            raise ValueError

        r = .5 * self.width / n if dim == 'w' else .5 * self.height / n     # радиус частицы
        dw, dh = 2 * r, r * np.sqrt(3.)                                     # шаг координат по ширине и высоте
        nw, nh = int(self.width / dw), int(self.height / dh)                # кол-во частиц по ширине и высоте

        parts = []
        for i in range(nh):         # вверх по стенке
            for j in range(nw):     # поперёк стенки
                h = i * dh
                w = j * dw if i % 2 == 0 else j * dw + r
                parts.append([w, h])
        self.particles = np.array(parts, dtype=np.float64)
        self.particles += self.pos

        if center:
            self.particles[:, 1] -= .5 * self.height

    def get_draw_particles(self, scale: np.ndarray, win_size: Tuple[int, int]) -> np.ndarray:
        """Преобразование физических координат в экранные координаты."""
        draw_parts = self.particles.copy()
        draw_parts[:, 1] += (win_size[1] // 2) / scale[1]   # центрирование на экране по оси Ox
        return draw_parts * scale

    def show_particles(self):
        plt.figure("Body", figsize=(6, 6))
        x, y = self.particles[:, 0], self.particles[:, 1]
        plt.scatter(x, y, color='k', marker='.')
        plt.title(self.name)
        plt.xlabel("ширина, м")
        plt.ylabel("высота, м")
        plt.show()


if __name__ == '__main__':
    print(Body(1, (100, 300), 'somebody', (255, 0, 0), np.array([20, 0])))
