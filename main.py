from typing import Tuple
import numpy as np

from visualizer import Visualizer
from body import Body
from space import Space
from solver import Solver


def main():
    make_some_data_for_report()     # для начальных картинок в отчёт
    modeling()

    return 0


def make_some_data_for_report():
    """Вспомогательная функция чисто для отчёта."""
    print("\nВспомогательная функция 'make_some_data_for_report'...")

    # Два тела - стенка и ударник
    wall = Body(mass=5, size=(10, 30), name='wall', color=(128, 128, 128), pos=np.array([20, 0]))
    striker = Body(mass=1, size=(10, 3), name='striker', color=(0, 0, 0), pos=np.array([0, 0]), rotate_deg=30)
    # Контрольный вывод
    print(wall, striker, sep='\n')
    # Разбивка тел на частицы
    wall.break_into_particles(n=25, dim='w', kind='wall')
    striker.break_into_particles(n=21, dim='h', kind='striker')
    # Вывод количества частиц в консоль
    print_n_particles(wall)
    print_n_particles(striker)
    # Показ тел
    wall.save_image()
    striker.save_image()


def print_n_particles(b: Body):
    print(f"Количество частиц в '{b.name}': {len(b.particles)}")


def modeling():
    """Основная функция программы. Запускает алгоритм моделирования."""
    print("\nМоделирование...")

    space = init_space()
    wall, striker = init_bodies()

    # Основная часть моделирования
    solver = Solver(wall, striker, space, sigma=.05, epsilon=1000)
    solver.build_mesh()

    Visualizer(solver, win_size=(900, 900)).show_static()   # отрисовка начального состояния


def init_bodies() -> Tuple[Body, Body]:
    """Инициализация тел и разбиение их на частицы."""
    wall = Body(mass=.5, size=(.35, 2.5), name='wall', color=(128, 128, 128), pos=np.array([.51, 0]))
    wall.break_into_particles(n=25, dim='w', kind='wall')
    print_n_particles(wall)

    striker = Body(mass=.1, size=(.5, .075), name='striker', color=(0, 0, 0), pos=np.array([0, 0]), rotate_deg=30)
    striker.break_into_particles(n=7, dim='h', kind='striker')
    print_n_particles(striker)

    return wall, striker


def init_space() -> Space:
    """Инициализация физического пространства."""
    return Space((3, 3))


if __name__ == '__main__':
    main()
