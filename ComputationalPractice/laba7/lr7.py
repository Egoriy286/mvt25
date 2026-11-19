from dolfin import *
import gmsh
import meshio
from pathlib import Path


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
        # Плоскости через A могут проходить внутри omega.
        return near(x[self.axis], self.value, 1e-10)


folder = Path(__file__).resolve().parent
results = folder / "results"
results.mkdir(exist_ok=True)

for name in ("omega", "omega_boundary", "omega_cells", "cells"):
    (results / name).mkdir(exist_ok=True)

set_log_level(LogLevel.ERROR)

# 1. Генерируем трёхмерную сетку.
gmsh.initialize()
try:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(folder / "mesh.geo"))
    n = int(gmsh.parser.getNumber("n")[0])
    gmsh.model.mesh.generate(3)
    gmsh.write(str(results / "mesh.msh"))
finally:
    gmsh.finalize()

# 2. Переносим тетраэдры и все три координаты в DOLFIN.
source = meshio.read(results / "mesh.msh")
tetra = source.get_cells_type("tetra")

if len(tetra) == 0:
    raise ValueError("Сетка не содержит тетраэдров.")

meshio.write(
    results / "mesh.xml",
    meshio.Mesh(source.points, [("tetra", tetra)]),
    file_format="dolfin-xml",
)

mesh = Mesh(str(results / "mesh.xml"))
h = 1.0 / n

# 3. Маркируем грубые кубы: размерность объёмных элементов — 3.
domains = MeshFunction("size_t", mesh, 3, 0)

for j in range(n):
    for i in range(n):
        for k in range(n):
            number = 1 + i + n * j + n * n * k
            Cube(i * h, j * h, k * h, h).mark(domains, number)

File(str(results / "domains.pvd")) << domains

# 4. Файлы подобластей и их меток.
submeshes = []
omega = File(str(results / "omega/omega.pvd"))
omega_boundary = File(str(results / "omega_boundary/omega_boundary.pvd"))
omega_cells = File(str(results / "omega_cells/omega_cells.pvd"))
cell_file = File(str(results / "cells/cells.pvd"))
boundary_file = File(str(results / "cells/cell_boundaries.pvd"))

# Внутри области у вершины 8 соседних кубов,
# на грани — 4, на ребре — 2, в углу — 1.
for j in range(n + 1):
    for i in range(n + 1):
        for k in range(n + 1):
            ax, ay, az = i * h, j * h, k * h

            omega_i = SubMesh(
                mesh,
                Cube(ax - h, ay - h, az - h, 2 * h),
            )

            # Размерность граней — 2.
            # Внешняя граница = 6; плоскости через A = 1, 2, 3.
            patch_boundary = MeshFunction("size_t", omega_i, 2, 0)
            patch_boundary.rename("omega_boundary", "Граница подобласти")

            DomainBoundary().mark(patch_boundary, 4)
            Side(0, ax).mark(patch_boundary, 1)
            Side(1, ay).mark(patch_boundary, 2)
            Side(2, az).mark(patch_boundary, 3)

            omega << omega_i
            omega_boundary << patch_boundary

            # 5. Маркируем соседние объёмные ячейки числами 1–8.
            cell_marks = MeshFunction("size_t", omega_i, 3, 0)
            cell_marks.rename("cell_id", "Номер составной ячейки")
            patch_cells = []

            for dz in range(2):
                for dy in range(2):
                    for dx in range(2):
                        cx = i + dx - 1
                        cy = j + dy - 1
                        cz = k + dz - 1

                        if not (
                            0 <= cx < n
                            and 0 <= cy < n
                            and 0 <= cz < n
                        ):
                            continue

                        number = 1 + dx + 2 * dy + 4 * dz
                        cube = Cube(cx * h, cy * h, cz * h, h)

                        cube.mark(cell_marks, number)
                        cell_mesh = SubMesh(omega_i, cube)

                        # 6. Маркируем грани отдельной ячейки.
                        boundaries = MeshFunction("size_t", cell_mesh, 2, 0)
                        boundaries.rename("boundary_id", "Метка границы ячейки")

                        DomainBoundary().mark(boundaries, 4)
                        Side(0, ax).mark(boundaries, 1)
                        Side(1, ay).mark(boundaries, 2)
                        Side(2, az).mark(boundaries, 3)

                        cell_file << cell_mesh
                        boundary_file << boundaries
                        patch_cells.append((cell_mesh, boundaries))

            omega_cells << cell_marks
            submeshes.append(
                (omega_i, patch_boundary, cell_marks, patch_cells)
            )

print("Подобластей:", len(submeshes))
print("Составных ячеек:", sum(len(item[3]) for item in submeshes))
print("Результаты:", results)
