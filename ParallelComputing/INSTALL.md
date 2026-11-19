# Установка MPI и запуск практических работ

Для сборки работ на C и C++ нужны компилятор и реализация MPI. Ниже приведены команды для Open MPI. Для работ с DOLFIN в репозитории подготовлен контейнер FEniCS Legacy со своим MPI.

## macOS

Установите инструменты командной строки Xcode, если они ещё отсутствуют. [Документация Apple](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/)

```bash
xcode-select --install
```

Дождитесь завершения установки, установите Homebrew и выполните предложенную установщиком команду для добавления `brew` в `PATH`. Затем установите Open MPI. [Установка Homebrew](https://brew.sh/)

```bash
brew install open-mpi
```

[Пакет Open MPI в Homebrew](https://formulae.brew.sh/formula/open-mpi)

## Ubuntu и Debian

```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev
```

[Пакеты Open MPI в Ubuntu](https://packages.ubuntu.com/noble/libopenmpi-dev) · [Пакеты Open MPI в Debian](https://packages.debian.org/source/trixie/openmpi)

## Windows через WSL

В PowerShell от имени администратора установите WSL с Ubuntu и перезагрузите компьютер. [Инструкция Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install)

```powershell
wsl --install -d Ubuntu
```

Откройте терминал Ubuntu и выполните команды из раздела [Ubuntu и Debian](#ubuntu-и-debian). Собирайте и запускайте программы в терминале Ubuntu.

## Проверка установки

```bash
command -v mpicc mpicxx mpirun
mpirun --version
```

Из корня репозитория соберите и запустите работу 1 на двух процессах.

```bash
cd ParallelComputing/laba1
mpicc laba1.c -o laba1
mpirun -n 2 ./laba1
```

Процесс 1 выведет сообщение, полученное от процесса 0. Используйте `mpicc`, `mpicxx` и `mpirun` из одной установки MPI.

Если Open MPI сообщает о нехватке слотов, уменьшите число процессов или добавьте `--oversubscribe` для проверки на одном компьютере. [Руководство Open MPI](https://docs.open-mpi.org/en/v5.0.4/man-openmpi/man1/mpirun.1.html)

## Запуск работ с FEniCS Legacy

Запустите Docker и выполните из корня репозитория.

```bash
docker compose up -d --build
docker compose exec -T fenics-legacy mpirun -n 2 python3 scripts/check_fenics.py
```

В контейнере используется MPICH, а образ в `compose.yaml` рассчитан на `linux/arm64`. Для запуска работ откройте терминал контейнера.

```bash
docker compose exec fenics-legacy bash
cd /workspace/fenics-legacy/ParallelComputing/laba4
mpirun -n 1 python3 laba4.py
```

[Команды для остальных работ](README.md) · [Подключение к контейнеру из VS Code](../FENICS.md)
