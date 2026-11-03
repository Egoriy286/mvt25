# Задание 3. Градиентный спуск

Открыть **[laba3.ipynb](laba3.ipynb)** и выполнить ячейки сверху вниз.

Выбран вариант 3: exp(x) на [−1,1], шаг 0.1, ε=0.0001.
Сначала вычисляются значения ряда Тейлора, затем обычный SGD подбирает коэффициенты полинома. Функции `forward`, `loss`, `gradient` и цикл обучения построены по примеру задания.

График — [results/gradient_descent.png](results/gradient_descent.png), веса — [results/weights.csv](results/weights.csv).

Код без notebook: [laba3.py](laba3.py). Запускать из папки этой работы, в контейнере: `python3 laba3.py`.
[Исходные материалы](materials/README.md).
