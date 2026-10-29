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

# Маленькая задача прогревает компиляцию форм и решатель до замера времени.
mesh = UnitSquareMesh(comm, 4, 4)
V = FunctionSpace(mesh, 'P', 1)
u, v = TrialFunction(V), TestFunction(V)
bc = DirichletBC(V, Constant(0.), 'on_boundary')
A, b = assemble_system(inner(grad(u),grad(v))*dx, Constant(1.)*v*dx, bc)
u = Function(V)
solve(A, u.vector(), b, 'cg', 'hypre_amg')

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

# Проверка невязки и энергетического тождества.
residual = b.copy()
A.mult(u.vector(), residual)
residual.axpy(-1., b)
error = residual.norm('l2') / b.norm('l2')
integral = assemble(u*dx)
energy = assemble(inner(grad(u),grad(u))*dx)
assert error < 1e-7 and abs(integral-energy) < 1e-8

if rank == 0:
    print('Процессов:', size)
    print('Вершин:', V.dim())
    print('Время:', elapsed, 'с')
    print('Относительная невязка:', error)
    print('Интеграл решения:', integral)
    path = Path('results/times.csv')
    new = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['processes','vertices','time_s','residual','integral'])
        writer.writerow([size,V.dim(),elapsed,error,integral])

# Вывод идёт после замера. Сохраняем общее решение на одном процессе.
if size == 1:
    File('results/solution.pvd') << u
    import numpy as np
    import matplotlib.pyplot as plt
    xy = mesh.coordinates()
    values = u.compute_vertex_values(mesh)
    np.savez('results/poisson.npz', xy=xy, cells=mesh.cells(), u=values)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.tricontourf(xy[:,0],xy[:,1],mesh.cells(),values,levels=20)
    fig.colorbar(im, ax=ax, label='u')
    ax.set(xlabel='x',ylabel='y',title='Уравнение Пуассона',aspect='equal')
    plt.tight_layout()
    plt.savefig('results/solution.png',dpi=150)
    plt.close(fig)
