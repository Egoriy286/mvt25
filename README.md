# mvt25

Учебные работы. Рабочая папка: `/Users/egoriy/Desktop/mvt25`.

## Что открыть

| Дисциплина | Задание | Основной файл |
|---|---|---|
| Искусственный интеллект | 3. Градиентный спуск | [laba3.ipynb](ArtificialIntelligence/laba3/laba3.ipynb) |
| Вычислительный практикум | 4. Transfinite 2D | [lr4.py](ComputationalPractice/laba4/lr4.py), [mesh.geo](ComputationalPractice/laba4/mesh.geo) |
| Вычислительный практикум | 5. Transfinite 3D | [lr5.py](ComputationalPractice/laba5/lr5.py), [mesh.geo](ComputationalPractice/laba5/mesh.geo) |
| Параллельные вычисления | 4. DOLFIN и MPI | [laba4.py](ParallelComputing/laba4/laba4.py), [запустить замеры](ParallelComputing/laba4/run.py) |
| Методы прикладной математики | 2. Газовая динамика | [laba2.ipynb](AppliedMathematics/laba2/laba2.ipynb) |
| Методы прикладной математики | 4. Теплопроводность | [laba4.ipynb](AppliedMathematics/laba4/laba4.ipynb) |
| Моделирование месторождений | 1. Фильтрация | [laba1.ipynb](HydrocarbonModeling/laba1/laba1.ipynb), [laba1.py](HydrocarbonModeling/laba1/laba1.py) |

В notebook всё идёт по порядку: задание → параметры → метод → расчёт → графики → вывод.
Параметры меняются в первой ячейке. Рядом лежит `.py` с тем же расчётом.
В сеточных работах геометрия и Python-файл лежат рядом, как в работах 1–3.

## Как запустить

1. Открыть эту папку в VS Code и выбрать **Dev Containers: Reopen in Container**.
2. Открыть нужный notebook из таблицы, выбрать ядро **Python (FEniCS Legacy)** и нажать **Run All**.
3. Для `.py` открыть файл и нажать `Cmd+Shift+B`: запуск идёт из папки файла. Для MPI команды указаны в [README работы](ParallelComputing/laba4/README.md).

Результаты каждой работы — в её папке `results/`. Файлы `.pvd` открываются в ParaView.
Оригинальные задания и твои notebooks — в `materials/`.

[Настройка Docker](FENICS.md) · [План работ](PLAN.md) · [Проверки](VERIFICATION.md) · [Источники](SOURCES.md).
