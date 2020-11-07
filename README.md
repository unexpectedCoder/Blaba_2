# Метод динамики частиц

Задача состоит в следующем.

По материалам, данным на лекции, нужно написать простую программу,
в которой одной группой частиц моделируется препятствие (вертикальная *стенка*),
второй группой частиц моделируется *ударник*. Форма ударника: шар или снаряд.
Картинка того, как это должно примерно выглядеть, приведена в лекции.

Для простоты:
- **потенциал типа Леннарда-Джонса**: потенциалы для обеих групп частиц могут быть одинаковы;
- **никакого количественного соответствия не требуется**:
только качественная картинка столкновения и разлета частиц,
которая более или менее похожа на ту, что была в лекции;
- **граничные условия**: если частица вылетает за границы расчетной области, она исчезает;
- **метод интегрирования**: один из приведенных на лекции.

Количество частиц должно определяться максимально возможной производительностью
(т.е. чем больше, тем лучше).

_**Для  убыстрения рекомендуется использовать метод сеток, рассмотренный на лекции!**_
