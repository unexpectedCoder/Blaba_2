from typing import Tuple
from pygame.locals import *
import pygame
import numpy as np

from body import Body
from cell import Cell
from solver import Solver


L_BLUE = 225, 225, 255
WHITE = 255, 255, 255


class Visualizer:
    def __init__(self, solver: Solver, win_size: Tuple[int, int] = (1200, 600)):
        self.wall = solver.wall.copy()
        self.striker = solver.striker.copy()
        self.mesh = solver.mesh
        self._win_size = win_size
        self._scale = np.array([win_size[0] / solver.space.size[0], win_size[1] / solver.space.size[1]])

        pygame.init()
        self.DISPLAYSURF = pygame.display.set_mode(win_size)
        pygame.display.set_caption("Particle Dynamics Method")
        self._fps = 1
        self._fps_clock = pygame.time.Clock()

    def show_static(self):
        self.draw()
        pygame.display.update()

        isRun = True
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    isRun = False

            if not isRun:
                break

    def draw(self):
        """Метод рисования объектов на экране."""
        self.DISPLAYSURF.fill(WHITE)
        for row in self.mesh.cells:
            for cell in row:
                rect = self.get_draw_rect(cell)
                pygame.draw.rect(self.DISPLAYSURF, L_BLUE, rect, width=1)
        for p, dp in zip(self.wall.particles, self.get_draw_particles(self.wall)):
            pygame.draw.circle(self.DISPLAYSURF, p.color, (dp[0], dp[1]), 1.5)
        for p, dp in zip(self.striker.particles, self.get_draw_particles(self.striker)):
            pygame.draw.circle(self.DISPLAYSURF, p.color, (dp[0], dp[1]), 1.5)

    def get_draw_particles(self, b: Body) -> np.ndarray:
        """Преобразовать физические координаты частиц в экранные координаты.

        :param b: тело, координаты частиц которого нужно преобразовать в экранные.
        :return: Экранные координаты частиц тела *b*.
        """
        draw_parts = np.array([p.pos.copy() for p in b.particles], order='F')
        draw_parts[:, 1] *= -1
        draw_parts[:, 1] += (self._win_size[1] // 2) / self._scale[1]   # центрирование на экране относительно оси Ox
        return draw_parts * self._scale

    def get_draw_rect(self, cell: Cell) -> Tuple[int, int, int, int]:
        """Преобразовать физические координаты ячейки сетки в экранные координаты.

        :param cell: ячейкка сетки.
        :return: Кортеж параметров прямоугольника типа ``pygame.rect.Rect``.
        """
        size = cell.size * self._scale
        dl = cell.dl.copy()
        dl[1] *= -1
        dl[1] += (self._win_size[1] // 2) / self._scale[1]           # центрирование на экране относительно оси Ox
        dl *= self._scale
        return dl[0], dl[1], size[0], size[1]
