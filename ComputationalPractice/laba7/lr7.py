from dolfin import *
import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent
results = folder / "results"
results.mkdir(exist_ok=True)
set_log_level(LogLevel.ERROR)

# gmsh.initialize()
# gmsh.option.setNumber("General.Terminal", 0)
# gmsh.open(str(folder / "mesh.geo"))
# n = int(gmsh.parser.getNumber("n")[0])
# gmsh.model.mesh.generate(2)
# gmsh.write(str(results / "mesh.msh"))
# gmsh.finalize()
n = 10
source = meshio.read(results / "mesh.msh")
tetra = source.get_cells_type("tetra")
meshio.write(
    results / "mesh.xml",
    meshio.Mesh(source.points[:, :2], [("tetra", tetra)]),
    file_format="dolfin-xml",
)
mesh = Mesh(str(results / "mesh.xml"))
h = 1.0 / n  # Сторона грубой ячейки.


class Cube(SubDomain):
    def __init__(self, ax, ay, az, h):
        super().__init__()
        self.ax, self.ay, self.az, self.h = ax, ay, az, h

    def inside(self, x, on_boundary):
        eps = 1e-10
        return (
            self.ax - eps <= x[0] <= self.ax + self.h + eps
            and self.ay - eps <= x[1] <= self.ay + self.h + eps
            and self.az - eps <= x[2] <= self.az + self.h + eps
        )


class Side(SubDomain):
    def __init__(self, axis, value):
        super().__init__()
        self.axis, self.value = axis, value

    def inside(self, x, on_boundary):
        return near(x[self.axis], self.value, 1e-10)


domains = MeshFunction("size_t", mesh, 2, 0)
for j in range(n):
    for i in range(n):
        for k in range(n):
            Cube(i * h, j * h, k * h, h).mark(domains, 1 + i + n * j + n * n * k)

File(str(results / "domains.pvd")) << domains

submeshes = []
omega = File(str(results / "omega/omega.pvd"))
omega_boundary = File(str(results / "omega_boundary/omega_boundary.pvd"))
omega_cells = File(str(results / "omega_cells/omega_cells.pvd"))
cell_file = File(str(results / "cells/cells.pvd"))
boundary_file = File(str(results / "cells/cell_boundaries.pvd"))
# Берём все вершины, включая стороны и углы области.
for j in range(n + 1):
    for i in range(n + 1):
        for k in range(n + 1):
            ax, ay, az = i * h, j * h, k * h
            omega_i = SubMesh(mesh, Cube(ax - h, ay - h, az - h, 2 * h))

            # Внешний контур = 6, координата по x A = 1, координата по y A = 2, координата по z A = 3.
            patch_boundary = MeshFunction("size_t", omega_i, 2, 0)
            patch_boundary.rename("omega_boundary", "Граница подобласти")
            Cube(ax - h, ay - h, az - h, 2 * h).mark(patch_boundary, 6)
            Side(0, ax).mark(patch_boundary, 1)
            Side(1, ay).mark(patch_boundary, 2)
            Side(2, az).mark(patch_boundary, 3)

            omega << omega_i
            omega_boundary << patch_boundary
            # 5. Выделяем соседние ячейки: внутри 4, по dx 1, по dy 2, по dz 3.

            cell_marks = MeshFunction("size_t", omega_i, 2, 0)
            cell_marks.rename("cell_id", "Номер составной ячейки")
            patch_cells = []
            for dz in range(2):
                for dy in range(2):
                    for dx in range(2):
                        cx, cy, cz = i + dx - 1, j + dy - 1, k + dz - 1
                        if not (0 <= cx < n and 0 <= cy < n and 0 <= cz < n):
                            continue  # За пределами области ячейки нет.
                        number = 1 + dx + 2 * dy + 4 * dz

                        cube = Cube(cx * h, cy * h, cz * h, h)
                        cube.mark(cell_marks, number)
                        cell_mesh = SubMesh(omega_i, cube)

                        boundaries = MeshFunction("size_t", cell_mesh, 1, 0)
                        boundaries.rename("boundary_id", "Метка границы ячейки")
                        DomainBoundary().mark(boundaries, 4)
                        Side(0, ax).mark(boundaries, 1)  # x сторона через A.
                        Side(1, ay).mark(boundaries, 2)  # y сторона через A.
                        Side(2, az).mark(boundaries, 3)  # z сторона через A.

                        cell_file << cell_mesh
                        boundary_file << boundaries
                        patch_cells.append((cell_mesh, boundaries))

            omega_cells << cell_marks
            submeshes.append((omega_i, patch_boundary, cell_marks, patch_cells))
