from body import Body
from space import Space


class Solver:
    """Класс решателя уравнений динамики частиц."""

    def __init__(self, wall: Body, striker: Body, space: Space,
                 sigma: float, epsilon: float):
        self.check_bodies_particles(wall, striker)

        self._wall = wall
        self._striker = striker
        self._space = space
        self._sig = sigma
        self._eps = epsilon

    @staticmethod
    def check_bodies_particles(w: Body, s: Body):
        if w.particles is None or s.particles is None:
            raise ValueError("тела не разбиты на частицы!")

    @property
    def sigma(self) -> float:
        return self._sig

    @property
    def epsilon(self) -> float:
        return self._eps
