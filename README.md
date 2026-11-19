# mvt25

Лабораторные и практические работы по численным методам, параллельным вычислениям и машинному обучению. В репозитории собраны исходники, ноутбуки Jupyter, условия задач и результаты расчётов.

## Дисциплины

| Раздел | Работы |
| --- | --- |
| [Методы решения задач прикладной математики](AppliedMathematics/README.md) | [Газовая динамика](AppliedMathematics/laba2/README.md) (конечные разности и МКЭ на подвижной сетке), [граничные условия](AppliedMathematics/laba3/README.md), [теплопроводность](AppliedMathematics/laba4/README.md), [теплопроводность с переносом](AppliedMathematics/laba5/README.md), [метод характеристик](AppliedMathematics/laba6/README.md) |
| [Параллельные вычисления](ParallelComputing/README.md) | [Замеры DOLFIN и MPI](ParallelComputing/laba4/README.md), [метод Якоби](ParallelComputing/laba6/README.md), [сравнение конечных разностей и МКЭ](ParallelComputing/laba7/README.md) |
| [Вычислительный практикум](ComputationalPractice/README.md) | [Сетки Transfinite 2D](ComputationalPractice/laba4/README.md) и [3D](ComputationalPractice/laba5/README.md), [подобласти вокруг вершин](ComputationalPractice/laba6/README.md) |
| [Моделирование углеводородных месторождений](HydrocarbonModeling/README.md) | [Фильтрация](HydrocarbonModeling/laba1/README.md), [смешанная задача фильтрации](HydrocarbonModeling/laba2/README.md) |
| [Искусственный интеллект](ArtificialIntelligence/README.md) | [Наивный Байес](ArtificialIntelligence/Наивный_Байес_Egory.ipynb), [градиентный спуск](ArtificialIntelligence/laba3/README.md) |

## Быстрый старт

Команды ниже выполняются из корня репозитория.

### Python и Jupyter

Создайте окружение Python 3.12 и запустите JupyterLab. Версии зависимостей закреплены в `uv.lock`. [Установка uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked
uv run jupyter lab
```

В JupyterLab откройте ноутбук выбранной работы и выполните **Run All**. В VS Code для локальных расчётов используйте интерпретатор из `.venv`.

### C/C++ и MPI

Установите компилятор и MPI по [инструкции](ParallelComputing/INSTALL.md). Для первого запуска соберите лабораторную № 1 и передайте сообщение по цепочке из четырёх процессов.

```bash
mpicc -O2 ParallelComputing/laba1/laba1.c -o /tmp/mvt25-laba1
mpirun -n 4 /tmp/mvt25-laba1
```

Команды для остальных работ собраны в [разделе параллельных вычислений](ParallelComputing/README.md).

### FEniCS Legacy и Gmsh

Для работ с `dolfin` и его Python-окружением используйте Docker. Конфигурация проекта рассчитана на `linux/arm64`.

```bash
docker compose up -d --build
docker compose exec -T fenics-legacy python3 scripts/check_fenics.py
```

В VS Code доступен запуск через **Dev Containers: Reopen in Container** с ядром **Python (FEniCS Legacy)**. [Настройка FEniCS, MPI и отладки](FENICS.md)

## Материалы и результаты

Условия и исходные примеры хранятся рядом с работами, часто в `materials/`. Графики и данные расчётов обычно сохраняются в `results/`. Файлы `.pvd` можно открыть в ParaView.

[Источники](SOURCES.md) · [Проверки](VERIFICATION.md) · [План работ](PLAN.md) · [Лицензия MIT](LICENSE)
