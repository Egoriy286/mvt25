"""Check the Legacy FEniCS environment with an exact linear Poisson solution.

Run inside the development container, optionally with mpirun -n 2.
No project files or previous calculation results are modified.
"""

import platform
import sys

import debugpy
import dolfin as df
import gmsh
import ipykernel
import meshio
import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    mesh = df.UnitSquareMesh(df.MPI.comm_world, 8, 8)
    space = df.FunctionSpace(mesh, "Lagrange", 1)
    exact = df.Expression("x[0] + 2*x[1]", degree=1)
    bc = df.DirichletBC(space, exact, "on_boundary")
    u = df.TrialFunction(space)
    v = df.TestFunction(space)
    solution = df.Function(space)
    df.solve(
        df.inner(df.grad(u), df.grad(v)) * df.dx
        == df.Constant(0.0) * v * df.dx,
        solution,
        bc,
    )
    l2_error = df.errornorm(exact, solution, "L2", degree_rise=1)
    if not np.isfinite(l2_error) or l2_error > 1e-10:
        raise RuntimeError("Poisson check failed: L2 error = {}".format(l2_error))
    if comm.rank == 0:
        print("Python: {} ({})".format(sys.version.split()[0], sys.executable))
        print("Architecture: {}".format(platform.machine()))
        print("DOLFIN: {}".format(df.__version__))
        print("Gmsh: {}".format(gmsh.__version__))
        print("MPI ranks: {}".format(comm.size))
        print("Jupyter kernel: {}; debugpy: {}".format(
            ipykernel.__version__, debugpy.__version__))
        print("meshio: {}".format(meshio.__version__))
        print("Poisson L2 error: {:.3e}".format(l2_error))
        print("FEniCS Legacy: OK")


if __name__ == "__main__":
    main()
