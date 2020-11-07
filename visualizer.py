from typing import Tuple
from pygame.locals import *
import pygame
import numpy as np


BLACK = 0, 0, 0
GREY = 128, 128, 128
WHITE = 255, 255, 255


class Visualizer:
    def __init__(self, wall: np.ndarray, body: np.ndarray,
                 wall_pos: np.ndarray, body_pos: np.ndarray,
                 real_size: Tuple[float, float], px_size: Tuple[int, int]):
        self._scale = np.array([px_size[0] / real_size[0], px_size[1] / real_size[1]])

        self._wall = wall.copy()
        self._wall += wall_pos / self._scale
        self._wall *= self._scale

        self._body = body.copy()
        self._body += body_pos / self._scale
        self._body *= self._scale

        pygame.init()
        self.DISPLAYSURF = pygame.display.set_mode(px_size)
        pygame.display.set_caption("Метод Динамики Частиц")
        self._fps = 1
        self._fps_clock = pygame.time.Clock()

    def showStatic(self):
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
        for p in self._wall:
            pygame.draw.circle(self.DISPLAYSURF, GREY, (p[0], p[1]), 1)
        for p in self._body:
            pygame.draw.circle(self.DISPLAYSURF, BLACK, (p[0], p[1]), 1)
