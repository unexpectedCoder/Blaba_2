from typing import Tuple
from pygame.locals import *
import pygame
import numpy as np

from body import Body
from space import Space


BLACK = 0, 0, 0
GREY = 128, 128, 128
WHITE = 255, 255, 255


class Visualizer:
    def __init__(self, wall: Body, striker: Body, space: Space,
                 win_size: Tuple[int, int] = (1200, 600)):
        self._wall = wall
        self._striker = striker
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
        self.DISPLAYSURF.fill(WHITE)
        for p in self._wall.get_draw_particles(self._scale, self._win_size):
            pygame.draw.circle(self.DISPLAYSURF, GREY, (p[0], p[1]), 1)
        for p in self._striker.get_draw_particles(self._scale, self._win_size):
            pygame.draw.circle(self.DISPLAYSURF, BLACK, (p[0], p[1]), 1)
