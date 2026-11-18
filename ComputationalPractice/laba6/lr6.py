from dolfin import *
import gmsh
import meshio
import matplotlib.pyplot as plt
from pathlib import Path


# 1. Два класса: квадрат и сторона границы.
class Square(SubDomain):
    def __init__(self, x, y, h):
        super().__init__()
        self.x, self.y, self.h = x, y, h

    def inside(self, x, on_boundary):
        eps = 1e-10
        return (
            self.x - eps <= x[0] <= self.x + self.h + eps
            and self.y - eps <= x[1] <= self.y + self.h + eps
        )


class Side(SubDomain):
    def __init__(self, axis, value):
        super().__init__()
        self.axis, self.value = axis, value

    def inside(self, x, on_boundary):
        # Линии через A могут лежать внутри omega, поэтому on_boundary не требуем.
        return near(x[self.axis], self.value, 1e-10)


# 2. Строим сетку Gmsh и читаем её в DOLFIN.
folder = Path(__file__).resolve().parent
results = folder / "results"
results.mkdir(exist_ok=True)
set_log_level(LogLevel.ERROR)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.open(str(folder / "mesh.geo"))
n = int(gmsh.parser.getNumber("n")[0])
gmsh.model.mesh.generate(2)
gmsh.write(str(results / "mesh.msh"))
gmsh.finalize()

source = meshio.read(results / "mesh.msh")
triangles = source.get_cells_type("triangle")
meshio.write(
    results / "mesh.xml",
    meshio.Mesh(source.points[:, :2], [("triangle", triangles)]),
    file_format="dolfin-xml",
)
mesh = Mesh(str(results / "mesh.xml"))
h = 1.0 / n  # Сторона грубой ячейки.


# 3. Отмечаем 100 грубых ячеек.
domains = MeshFunction("size_t", mesh, 2, 0)
for j in range(n):
    for i in range(n):
        Square(i * h, j * h, h).mark(domains, 1 + i + n * j)

File(str(results / "domains.pvd")) << domains


# 4. Общие файлы: каждый << добавляет следующий шаг времени.
submeshes = []
omega = File(str(results / "omega.pvd"))
omega_boundary = File(str(results / "omega_boundary.pvd"))
omega_cells = File(str(results / "omega_cells.pvd"))
cell_file = File(str(results / "cells/cells.pvd"))
boundary_file = File(str(results / "cells/cell_boundaries.pvd"))
# Берём все вершины, включая стороны и углы области.
for j in range(n + 1):
    for i in range(n + 1):
        ax, ay = i * h, j * h
        omega_i = SubMesh(mesh, Square(ax - h, ay - h, 2 * h))

        # Внешний контур = 3, вертикаль через A = 1, горизонталь = 2.
        patch_boundary = MeshFunction("size_t", omega_i, 1, 0)
        patch_boundary.rename("omega_boundary", "Граница подобласти")
        DomainBoundary().mark(patch_boundary, 3)
        Side(0, ax).mark(patch_boundary, 1)
        Side(1, ay).mark(patch_boundary, 2)
        omega << omega_i
        omega_boundary << patch_boundary

        # 5. Выделяем соседние ячейки: внутри 4, на стороне 2, в углу 1.
        cell_marks = MeshFunction("size_t", omega_i, 2, 0)
        cell_marks.rename("cell_id", "Номер составной ячейки")
        patch_cells = []
        for dy in range(2):
            for dx in range(2):
                cx, cy = i + dx - 1, j + dy - 1
                if not (0 <= cx < n and 0 <= cy < n):
                    continue  # За пределами области ячейки нет.
                number = 1 + dx + 2 * dy
                square = Square(cx * h, cy * h, h)
                square.mark(cell_marks, number)
                cell_mesh = SubMesh(omega_i, square)

                boundaries = MeshFunction("size_t", cell_mesh, 1, 0)
                boundaries.rename("boundary_id", "Метка границы ячейки")
                DomainBoundary().mark(boundaries, 3)
                Side(0, ax).mark(boundaries, 1)  # Вертикальная сторона через A.
                Side(1, ay).mark(boundaries, 2)  # Горизонтальная сторона через A.

                cell_file << cell_mesh
                boundary_file << boundaries
                patch_cells.append((cell_mesh, boundaries))

        omega_cells << cell_marks
        submeshes.append((omega_i, patch_boundary, cell_marks, patch_cells))


# 6. Рисунок сетки.
fig, ax = plt.subplots(figsize=(7, 7))
ax.triplot(
    mesh.coordinates()[:, 0],
    mesh.coordinates()[:, 1],
    mesh.cells(),
    color="black",
    lw=0.2,
)
for k in range(n + 1):
    ax.axvline(k * h, color="royalblue", lw=1)
    ax.axhline(k * h, color="royalblue", lw=1)
ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="x",
    ylabel="y",
    title="Сетка 1x1, грубые ячейки 10x10",
    aspect="equal",
)
fig.tight_layout()
fig.savefig(results / "mesh.png", dpi=150)
plt.close(fig)

print("Подобластей:", len(submeshes))
print("Составных ячеек:", sum(len(item[3]) for item in submeshes))
print("Результаты:", results)
