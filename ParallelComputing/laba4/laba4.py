from dolfin import *
from mpi4py import MPI
from pathlib import Path
import csv

# -div(grad(u)) = 1 на квадрате 1x1, u = 0 на всей границе.
N = 320
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_log_level(LogLevel.ERROR)
Path('results').mkdir(exist_ok=True)

# Основная задача.
comm.Barrier()
start = MPI.Wtime()
mesh = UnitSquareMesh(comm, N, N)
V = FunctionSpace(mesh, 'P', 1)
u, v = TrialFunction(V), TestFunction(V)
bc = DirichletBC(V, Constant(0.), 'on_boundary')
a = inner(grad(u),grad(v))*dx
l = Constant(1.)*v*dx

A, b = assemble_system(a, l, bc)

u = Function(V)
solver = KrylovSolver('cg', 'hypre_amg')
solver.parameters['relative_tolerance'] = 1e-10
solver.parameters['absolute_tolerance'] = 1e-12
solver.solve(A, u.vector(), b)
comm.Barrier()
elapsed = comm.allreduce(MPI.Wtime()-start, op=MPI.MAX)


if rank == 0:
    print('Процессов:', size)
    print('Вершин:', V.dim())
    print('Время:', elapsed, 'с')
    File('results/solution.pvd') << u