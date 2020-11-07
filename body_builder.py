import numpy as np


def build_body(w: float, h: float, n: int, key: str = 'w') -> np.ndarray:
    """Построить тело из частиц (гексагональная упаковка).

    :param w: ширина стенки, мм.
    :param h: высота стенки, мм.
    :param n: кол-во частиц по ширине или по высоте в зависимости от значения *key*.
    :param key: если 'w', то *n* означает кол-во частиц по ширине, если 'h' -- количество частиц по высоте.
    :return: Массив сгенерированных частиц.
    """
    if key != 'w' and key != 'h':
        raise ValueError

    r = .5 * w / n if key == 'w' else .5 * h / n    # радиус частицы
    dw, dh = 2 * r, r * np.sqrt(3.)                 # шаг координат по ширине и высоте
    nw, nh = int(w / dw), int(h / dh)               # кол-во частиц по ширине и высоте

    particles = []
    for i in range(nh):                             # вверх по стенке
        for j in range(nw):                         # поперёк стенки
            h = i * dh
            w = j * dw if i % 2 == 0 else j * dw + r
            particles.append([w, h])
    particles = np.array(particles, dtype=np.float64)
    particles[:, 1] -= .5*h                         # центрирование отностиельно оси Ox

    return particles
