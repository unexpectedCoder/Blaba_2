import numpy as np
import os
import pickle

from visualizer import Visualizer
from body import Body
from space import Space
from solver import Solver


def main():
    # make_some_data_for_report()     # для начальных картинок в отчёт
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


def modeling():
    """Основная функция программы. Запускает алгоритм моделирования."""
    print("\nМоделирование...")

    load = False
    if os.path.isdir('data'):
        if os.path.isfile('data/start_wall') and os.path.isfile('data/start_striker'):
            load = input(" > Загружать начальные данные из файлов или сгенерировать заново? (y/n) ")
            load = True if load == 'y' else False

    wall = init_wall(load)
    striker = init_striker(load)
    space = init_space()

    # Основная часть моделирования
    solver = Solver(wall, striker, space, sigma=.01, epsilon=5e3)
    solver.create_mesh(load)
    Visualizer(solver, win_size=(900, 900)).show_static()   # отрисовка начального состояния

    # solver.relax((0., 1.), 1e-3)
    # Visualizer(solver, win_size=(900, 900)).show_static()   # отрисовка конца релаксации
    #
    # v0 = np.array([1000, 0])
    # a = striker.rotate
    # v0 = np.matmul([[np.cos(a), np.sin(a)],
    #                 [-np.sin(a), np.cos(a)]], v0.T)
    # solver.solve((0, 0.1), 2e-5, v0, striker.name)
    # Visualizer(solver, win_size=(900, 900)).show_static()   # отрисовка конца моделирования


def init_wall(load: bool) -> Body:
    """Инициализировать объект *стенки*, разбив его на частицы.

    :return: Объект *стенки* типа ``Body``.
    """
    w = Body(mass=5., size=(.1, .75), name='wall', color=(128, 128, 128), pos=np.array([.2, 0]))

    if load:
        path = 'data/start_wall'
        with open(path, 'rb') as f:
            w.particles = pickle.load(f)
    else:
        w.break_into_particles(n=31, dim='w', kind='wall')
        w.save_particles()
    print_n_particles(w)

    return w


def init_striker(load: bool) -> Body:
    """Инициализировать объект *ударника*, разбив его на частицы.

    :return: Объект *ударника* типа ``Body``.
    """
    s = Body(mass=1., size=(.12, .012), name='striker', color=(0, 0, 0), pos=np.array([0.091, 0.]), rotate_deg=30)

    if load:
        path = 'data/start_striker'
        with open(path, 'rb') as f:
            s.particles = pickle.load(f)
    else:
        s.break_into_particles(n=15, dim='h', kind='striker')
        s.save_particles()
    print_n_particles(s)

    return s


def init_space() -> Space:
    """Инициализация физического пространства."""
    return Space((1, 1))


def print_n_particles(b: Body):
    print(f"Количество частиц в '{b.name}': {len(b.particles)}")


if __name__ == '__main__':
    main()
