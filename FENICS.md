# FEniCS Legacy в VS Code

## Окружение

- Контейнер и имя окружения: `fenics-legacy`.
- Локальная папка: `/Users/egoriy/Desktop/mvt25`.
- Та же папка в контейнере: `/workspace/fenics-legacy`.
- Linux ARM64: нативная архитектура Apple Silicon.
- База: `ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30`,
  закреплённая по digest в `.devcontainer/Dockerfile`.
- В базе DOLFIN `2019.2.0.dev0`; это Legacy API `import dolfin`,
  совместимость конкретных старых работ проверяется отдельно.
- Дополнительно установлены ipykernel `6.29.5`, debugpy `1.8.14`,
  nbformat `5.10.4` и nbclient `0.10.2`.

Расчётные зависимости устанавливаются в Docker-образ. Файлы проекта подключены
через bind mount: изменения из VS Code и контейнера сохраняются в одной
папке на рабочем столе. Дополнительные пакеты для повторяемого окружения
добавлять в `.devcontainer/Dockerfile`.

## Открыть проект

1. Запустить Docker Desktop.
2. Открыть `/Users/egoriy/Desktop/mvt25` в VS Code.
3. Нажать `Cmd+Shift+P` → **Dev Containers: Reopen in Container**.
4. Дождаться подключения; слева внизу появится **Dev Container: fenics-legacy**.

При открытии автоматически собирается локальный образ с кэшированием слоёв.
VS Code подключается к сервису из `compose.yaml`, использует контейнерный
`/usr/bin/python3` и устанавливает расширения Python/Jupyter в контейнер.

## Запуск и отладка

Открыть новый терминал VS Code после подключения. Он работает внутри Linux.

```bash
python3 scripts/check_fenics.py
```

Проверка импортирует DOLFIN, Gmsh, mpi4py, meshio, Jupyter и debugpy,
а затем решает задачу Пуассона с известным линейным решением.

Для старых работ переходить в папку самой работы, поскольку пути к сеткам
в них относительные:

```bash
cd /workspace/fenics-legacy/ComputationalPractice/laba1
python3 laba1.py
```

Для текущего Python-файла: `Cmd+Shift+B` запускает задачу из `.vscode/tasks.json`;
`F5` → **FEniCS Legacy: текущий Python-файл** запускает отладчик.
Оба способа используют папку открытого файла как рабочую директорию.
Эти задачи предназначены для окна, подключённого к контейнеру.

Для notebook выбрать ядро **Python (FEniCS Legacy)**; при необходимости
**Select Another Kernel → Jupyter Kernel**. Python из macOS выбирать не нужно.

## MPI

```bash
cd /workspace/fenics-legacy
mpirun -n 2 python3 scripts/check_fenics.py
```

В образе используется MPICH; дополнительный флаг Open MPI
`--allow-run-as-root` здесь не нужен и не поддерживается.
По умолчанию в Compose задано максимум 4 CPU и 4 ГиБ памяти;
для крупных экспериментов изменить `cpus` и `mem_limit`, учитывая ресурсы
Docker Desktop, и пересоздать контейнер. Замеры на числе рангов больше
выделенных CPU не интерпретировать как ускорение на отдельных ядрах.

## Управление из терминала macOS

```bash
cd /Users/egoriy/Desktop/mvt25
bash scripts/prepare_fenics.sh
docker compose up -d
docker exec -it fenics-legacy bash
```

Остановить вычислительное окружение, сохранив контейнер и файлы:

```bash
docker compose stop
```

При закрытии окна Dev Container сервис также останавливается согласно
настройке `shutdownAction`. Завершать Docker Desktop можно после остановки.

## Рисунки и сетки

`MPLBACKEND=Agg` позволяет сохранять рисунки без отдельного графического
сервера. Использовать `savefig(...)` либо встроенное отображение Jupyter.
Обычный `plt.show()` в скрипте не открывает окно macOS.
Gmsh доступен для генерации сеток; графическое окно Linux в macOS не настраивалось.

## Источники

- [Установка Legacy FEniCS](https://fenicsproject.org/download/archive/).
- [Работа в Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers).
- [Настройка окружения VS Code](https://code.visualstudio.com/docs/devcontainers/create-dev-container).

## Проверено 29.10.2026

- DOLFIN `2019.2.0.dev0`, Gmsh `4.12.2`, mpi4py `3.1.6`, Python `3.10.12`.
- Задача Пуассона: L2-ошибка `3.210e-16` на одном процессе,
  `2.801e-16` на двух MPI-процессах.
- Ядро `fenics-legacy` запускается и исполняет код с `import dolfin`.
- `pip check` не обнаружил нарушенных зависимостей.
- Проверена запись из контейнера в локальную папку репозитория.

Проверки лабораторных описаны в [VERIFICATION.md](VERIFICATION.md).
Открыть нужный notebook из [таблицы работ](README.md) и выбрать **Run All**.
Рабочая директория notebook — его папка; `.py` также запускается из своей папки.
Расчёты редактируются непосредственно в файлах работ, генератор не нужен.
