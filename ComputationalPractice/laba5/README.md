# Задание 5. Transfinite 3D

Основные файлы: **[lr5.py](lr5.py)** и **[mesh.geo](mesh.geo)**.

Порядок расчёта:

1. Gmsh строит куб 1×1×1, по 10 крупных блоков вдоль оси и по 4 мелких разбиения в блоке.
2. Recombine создаёт гексаэдры. Для `DOLFIN SubMesh` они делятся на 6 согласованных тетраэдров; исходная сетка Gmsh сохраняется в `mesh.msh`.
3. Крупные блоки выбираются по координатам центров элементов, без физических меток Gmsh.
4. `SubMesh` выделяет 1000 подсеток, `Boundary().mark(facets, 1)` отмечает границы.

В каждой подсетке 384 тетраэдра, мера 0.001, мера границы 0.06. Их сумма даёт единичную область.

Открыть в ParaView `results/blocks.pvd` для разбиения всей области. Отдельные подсетки и границы — `results/submeshes/submesh_0000.pvd`, `facets_0000.pvd` и т. д.
[Рисунок сетки](results/mesh.png) · [Таблица подсеток](results/submeshes.csv).

В терминале контейнера:

```bash
cd /workspace/fenics-legacy/ComputationalPractice/laba5
python3 lr5.py
```

Можно открыть `lr5.py` и нажать `Cmd+Shift+B`. [Исходное задание](materials/README.md).
