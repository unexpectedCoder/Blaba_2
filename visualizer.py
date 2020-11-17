from typing import Tuple
from pygame.locals import *
import pygame
import numpy as np

from solver import Solver


L_BLUE = 220, 220, 255
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
        self.DISPLAYSURF.fill(WHITE)
        for row in self.mesh.cells:
            for cell in row:
                rect = cell.as_draw_rect(self._scale, self._win_size)
                pygame.draw.rect(self.DISPLAYSURF, L_BLUE, rect, width=1)
        for p in self.wall.get_draw_particles(self._scale, self._win_size):
            pygame.draw.circle(self.DISPLAYSURF, self.wall.color, (p[0], p[1]), 1)
        for p in self.striker.get_draw_particles(self._scale, self._win_size):
            pygame.draw.circle(self.DISPLAYSURF, self.striker.color, (p[0], p[1]), 1)
