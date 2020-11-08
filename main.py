import matplotlib.pyplot as plt
import numpy as np

from visualizer import Visualizer
import body_builder as bb


def main():
    # Часть кода для картинок в отчёт
    wall = bb.build_body(w=100, h=300, n=50, key='w')
    showBody(wall, "Стенка")
    body = bb.build_body(w=500, h=25, n=10, key='h')
    showBody(body, "Тело (снаряд)")

    # Начало pygame
    wall = bb.build_body(w=300, h=600, n=150, key='w')
    body = bb.build_body(w=500, h=25, n=15, key='h')
    # Отрисовка начального состояния
    Visualizer(wall, body, np.array([505, 0]), np.array([0, 0]), (5000, 600),
               px_size=(1200, 600), offset=200).showStatic()

    return 0


def showBody(body: np.ndarray, title: str):
    """Отрисовка упаковок частиц в теле для контроля правильности и отладки.

    :param body: массив частиц тела.
    :param title: заголовок графика.
    """
    plt.figure("Body", figsize=(8, 8))
    plt.scatter(body[:, 0], body[:, 1], color='k', marker='.')
    plt.title(title)
    plt.xlabel("ширина, мм")
    plt.ylabel("высота, мм")
    plt.show()


if __name__ == '__main__':
    main()
