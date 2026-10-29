# Задание 4. Transfinite 2D

Основные файлы: **[lr4.py](lr4.py)** и **[mesh.geo](mesh.geo)**.

Порядок расчёта:

1. Gmsh строит квадрат 1×1, по 10 крупных блоков вдоль оси и по 4 мелких разбиения в блоке.
2. Recombine создаёт четырёхугольники. Для `DOLFIN SubMesh` они делятся на 2 треугольника; исходная сетка Gmsh сохраняется в `mesh.msh`.
3. Крупные блоки выбираются по координатам центров элементов, без физических меток Gmsh.
4. `SubMesh` выделяет 100 подсеток, `Boundary().mark(facets, 1)` отмечает границы.

В каждой подсетке 32 треугольника, мера 0.01, мера границы 0.4. Их сумма даёт единичную область.

Открыть в ParaView `results/blocks.pvd` для разбиения всей области. Отдельные подсетки и границы — `results/submeshes/submesh_0000.pvd`, `facets_0000.pvd` и т. д.
[Рисунок сетки](results/mesh.png) · [Таблица подсеток](results/submeshes.csv).

В терминале контейнера:

```bash
cd /workspace/fenics-legacy/ComputationalPractice/laba4
python3 lr4.py
```

Можно открыть `lr4.py` и нажать `Cmd+Shift+B`. [Исходное задание](materials/README.md).
