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


def modeling():
    """Основная функция программы. Запускает алгоритм моделирования."""
    print("\nМоделирование...")

    wall = init_wall()
    striker = init_striker()
    space = init_space()
    vis = Visualizer(space, win_size=(900, 900))

    # Основная часть моделирования
    # Начальное состояние
    solver = Solver(wall, striker, space,
                    sigma=0.003, epsilon=1e-10)
    solver.gen_mesh()
    vis.show_in_static(solver)
    # Релаксация
    # ...ударника
    striker = solver.relax_striker(5e-4, (0, 1e-2))
    vis.show_in_static(solver)
    solver.clear_mesh()
    # ...стенки
    wall = solver.relax_wall(1e-3, (0, 2.5e-2))
    vis.show_in_static(solver)
    solver.clear_mesh()
    # Основная симуляция
    a = striker.rotate
    v0 = np.array([800. * np.cos(a), 800. * np.sin(a)])
    solver.solve(wall.copy(), striker.copy(), 1e-6, (0, 0.2/800), v0)
    vis.show_in_static(solver)


def init_wall() -> Body:
    """Инициализировать объект *стенки*, разбив его на частицы.

    :return: Объект *стенки* типа ``Body``.
    """
    w = Body(mass=1.5, size=(.15, .75), name='wall', color=(128, 128, 128), pos=np.array([.3, 0]))
    w.break_into_particles(n=35, dim='w', kind='wall')
    print_n_particles(w)
    return w


def init_striker() -> Body:
    """Инициализировать объект *ударника*, разбив его на частицы.

    :return: Объект *ударника* типа ``Body``.
    """
    s = Body(mass=1., size=(.15, .025), name='striker', color=(0, 0, 0), pos=np.array([0.16, 0.]), rotate_deg=30)
    s.break_into_particles(n=15, dim='h', kind='striker')
    print_n_particles(s)
    return s


def init_space() -> Space:
    """Инициализация физического пространства."""
    return Space((1, 1))


def print_n_particles(b: Body):
    print(f"Количество частиц в '{b.name}': {len(b.particles)}")


if __name__ == '__main__':
    main()
