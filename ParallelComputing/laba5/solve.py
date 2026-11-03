from dolfin import *
from mpi4py import MPI
from pathlib import Path
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arg1", type=str, default="default")
parser.add_argument("--arg2", type=str, default="none")
args = parser.parse_args()

solver_name = args.arg1
precond_name = args.arg2
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

solver = KrylovSolver(solver_name, precond_name)
solver.parameters['relative_tolerance'] = 1e-10
solver.parameters['absolute_tolerance'] = 1e-12
solver.solve(A, u.vector(), b)
comm.Barrier()
elapsed = comm.allreduce(MPI.Wtime()-start, op=MPI.MAX)


if rank == 0:
    print(elapsed)