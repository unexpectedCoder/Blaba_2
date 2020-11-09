from typing import Tuple
import numpy as np

from visualizer import Visualizer
from body import Body


def main():
    make_some_data_for_report()     # для начальных картинок в отчёт
    modeling()

    return 0


def make_some_data_for_report():
    """Вспомогательная функция чисто для отчёта."""
    print("\nВспомогательная функция 'make_some_data_for_report'...")

    # Два тела - стенка и ударник
    wall = Body(mass=5, size=(10, 30), name='wall', color=(128, 128, 128), pos=np.array([20, 0]))
    striker = Body(mass=.5, size=(10, 3), name='striker', color=(0, 0, 0), pos=np.array([0, 0]))
    # Контрольный вывод
    print(wall, striker, sep='\n')
    # Разбивка тел на частицы
    wall.break_into_particles(n=25, dim='w')
    striker.break_into_particles(n=20, dim='h')
    # Вывод количества частиц в консоль
    print_n_particles(wall)
    print_n_particles(striker)
    # Показ тел
    wall.show_particles()
    striker.show_particles()


def print_n_particles(b: Body):
    print(f"Количество частиц в '{b.name}': {b.particles.size}")


def modeling():
    """Основная функция программы. Запускает алгоритм моделирования."""
    print("\nМоделирование...")

    wall, striker = init_bodies()
    Visualizer(wall, striker, space_size=(5, 3)).show_static()  # отрисовка начального состояния


def init_bodies() -> Tuple[Body, Body]:
    """Инициализация тел и разбиение их на частицы."""
    wall = Body(mass=.5, size=(.5, 3), name='wall', color=(128, 128, 128), pos=np.array([.55, 0]))
    wall.break_into_particles(n=175, dim='w')
    print_n_particles(wall)

    striker = Body(mass=.1, size=(.5, .075), name='striker', color=(0, 0, 0), pos=np.array([0, 0]))
    striker.break_into_particles(n=25, dim='h')
    print_n_particles(striker)

    return wall, striker


if __name__ == '__main__':
    main()
