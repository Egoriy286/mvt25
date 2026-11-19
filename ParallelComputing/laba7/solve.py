import argparse
import sys

import dolfin as df
import numpy as np
from mpi4py import MPI


class DifferenceSolution(df.UserExpression):
    def __init__(self, nodes, values, **kwargs):
        super().__init__(**kwargs)
        self.nodes = nodes
        self.values = values

    def eval(self, value, x):
        value[0] = np.interp(x[0], self.nodes, self.values)

    def value_shape(self):
        return ()


def assemble_poisson(comm, N):
    mesh = df.UnitIntervalMesh(comm, N)
    V = df.FunctionSpace(mesh, 'P', 1)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    bc = df.DirichletBC(V, df.Constant(0.0), 'on_boundary')
    a = df.Constant(0.1) * df.inner(df.grad(u), df.grad(v)) * df.dx
    l = df.Constant(1.0) * v * df.dx
    A, b = df.assemble_system(a, l, bc)
    return V, A, b


def main():
    parser = argparse.ArgumentParser(description='Задача Пуассона методом конечных элементов')
    parser.add_argument('N', type=int, help='число ячеек сетки')
    parser.add_argument('solution', help='CSV с разностным решением на узлах сетки')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if args.N < 2 or args.N <= size:
        parser.error('N должно быть не меньше 2 и больше числа процессов')
    df.set_log_level(df.LogLevel.ERROR)

    data, error = None, None
    if rank == 0:
        try:
            data = np.loadtxt(args.solution, delimiter=',', skiprows=1, ndmin=2)
            if (data.shape != (args.N + 1, 2) or not np.isfinite(data).all()
                    or not np.allclose(data[:, 0], np.linspace(0, 1, args.N + 1), rtol=0, atol=1e-12)
                    or data[0, 1] != 0 or data[-1, 1] != 0):
                raise ValueError('ожидаются N+1 строк со значениями (x,u) и нули на границе')
        except (OSError, ValueError) as exc:
            error = str(exc)
    data, error = comm.bcast((data, error), root=0)
    if error is not None:
        if rank == 0:
            print(f'Ошибка чтения {args.solution}: {error}', file=sys.stderr, flush=True)
        return 1

    # Прогрев компиляции форм перед замером.
    assemble_poisson(comm, max(8, 2 * size))

    comm.Barrier()
    start = MPI.Wtime()
    V, A, b = assemble_poisson(comm, args.N)
    u = df.Function(V)
    df.LUSolver(comm, 'mumps').solve(A, u.vector(), b)
    elapsed = comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)

    # Оба решения лежат в одном P1-пространстве.
    u_fd = df.interpolate(DifferenceSolution(data[:, 0], data[:, 1], degree=1), V)
    difference = df.Function(V)
    difference.vector().axpy(1.0, u_fd.vector())
    difference.vector().axpy(-1.0, u.vector())
    relative_error = df.norm(difference, 'L2') / df.norm(u, 'L2')
    if rank == 0:
        print(f'N={args.N} processes={size} time_s={elapsed:.9f} rel_l2={relative_error:.6e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
