from typing import Tuple
from pygame.locals import *
import pygame
import numpy as np

from space import Space
from cell import Cell
from solver import Solver


L_BLUE = 225, 225, 255
WHITE = 255, 255, 255


class Visualizer:
    def __init__(self, space: Space, solver: Solver, win_size: Tuple[int, int] = (1200, 600)):
        self.cells = solver.cells.copy()
        self._win_size = win_size
        self._scale = np.array([win_size[0] / space.size[0], win_size[1] / space.size[1]])

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

        # Сетка
        for row in self.cells:
            for c in row:
                rect = self.get_draw_rect(c)
                pygame.draw.rect(self.DISPLAYSURF, L_BLUE, rect, width=1)
        # Тела
        parts = []
        for row in self.cells:
            for c in row:
                parts += c.particles
        if parts:
            draw_parts = self.get_draw_particles(parts)
            for p, dp in zip(parts, draw_parts):
                pygame.draw.circle(self.DISPLAYSURF, p.color, (dp[0], dp[1]), 2)

    def get_draw_particles(self, parts) -> np.ndarray:
        draw_parts = np.array([p.pos for p in parts], order='F')
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
