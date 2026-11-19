import argparse
from pathlib import Path
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory


def measure(command):
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    for line in result.stdout.splitlines():
        if line.startswith('N='):
            return dict(field.split('=', 1) for field in line.split())
    raise RuntimeError(f'Сводка замера отсутствует в выводе:\n{result.stdout}')


def main():
    parser = argparse.ArgumentParser(description='Сравнение МКР (MPI + Якоби) и МКЭ (FEniCS)')
    parser.add_argument('--cells', type=int, default=128, help='число ячеек (по умолчанию 128)')
    parser.add_argument('--processes', type=int, nargs='+', default=[1, 2, 4],
                        help='число процессов для каждого замера (по умолчанию 1 2 4)')
    parser.add_argument('--repeats', type=int, default=3, help='число повторов (по умолчанию 3)')
    args = parser.parse_args()
    processes = sorted({1, *args.processes})
    if processes[0] < 1 or processes[-1] >= args.cells or args.repeats < 1:
        parser.error('требуется 1 ≤ число процессов < число ячеек и хотя бы один повтор')

    source_dir = Path(__file__).resolve().parent
    with TemporaryDirectory(prefix='laba7-') as tmp:
        binary = Path(tmp) / 'laba7'
        solution = Path(tmp) / 'solution.csv'
        subprocess.run(['mpicc', '-std=c11', '-O3', '-Wall', '-Wextra', '-Wpedantic',
                        str(source_dir / 'laba7.c'), '-lm', '-o', str(binary)], check=True)

        print(f'Ячеек: {args.cells}, замеров на процесс: {args.repeats}', flush=True)
        print('Процессы |  МКР, с  | S МКР |  МКЭ, с  | S МКЭ | Итерации | Невязка МКР | Отн. ошибка L2',
              flush=True)
        for p in processes:
            mpi = ['mpirun', '-n', str(p)]
            fd = []
            fem = []
            for _ in range(args.repeats):
                fd.append(measure([*mpi, str(binary), str(args.cells), str(solution)]))
                fem.append(measure([*mpi, sys.executable, str(source_dir / 'solve.py'),
                                    str(args.cells), str(solution)]))

            fd_time = median(float(row['time_s']) for row in fd)
            fem_time = median(float(row['time_s']) for row in fem)
            residual = max(float(row['residual']) for row in fd)
            error = max(float(row['rel_l2']) for row in fem)
            if p == 1:
                fd_base, fem_base = fd_time, fem_time
            print(f"{p:8d} | {fd_time:8.6f} | {fd_base / fd_time:5.3f} | "
                  f"{fem_time:8.6f} | {fem_base / fem_time:5.3f} | {fd[0]['iterations']:>8} | "
                  f"{residual:11.3e} | {error:14.3e}", flush=True)


if __name__ == '__main__':
    main()
