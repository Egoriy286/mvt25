from dolfin import *
import gmsh
import meshio
import matplotlib.pyplot as plt
from pathlib import Path


# Шаг 1. Классы для выделения областей и разметки границ.
class Rectangle(SubDomain):
    def __init__(self, xmin, xmax, ymin, ymax):
        super().__init__()
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def inside(self, x, on_boundary):
        eps = 1e-10
        return (
            self.xmin - eps <= x[0] <= self.xmax + eps
            and self.ymin - eps <= x[1] <= self.ymax + eps
        )


class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class VerticalBoundary(SubDomain):
    def __init__(self, ax):
        super().__init__()
        self.ax = ax

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self.ax, 1e-10)


class HorizontalBoundary(SubDomain):
    def __init__(self, ay):
        super().__init__()
        self.ay = ay

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], self.ay, 1e-10)


# Шаг 2. Строим в Gmsh квадрат 1x1 с грубой сеткой 10x10.
# Пути считаем от файла, поэтому запускать можно из любой папки.
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


# Шаг 3. Метки грубых ячеек: 1, 2, ..., 100.
# Все мелкие треугольники одного квадрата получают одну метку.
domains = MeshFunction("size_t", mesh, 2, 0)
for j in range(n):
    for i in range(n):
        number = 1 + i + n * j
        Rectangle(i * h, (i + 1) * h, j * h, (j + 1) * h).mark(domains, number)

File(str(results / "domains.pvd")) << domains


# Шаг 4. Вокруг каждой внутренней вершины A выделяем четыре квадрата.
# Подобласти перекрываются, поэтому для каждой создаём свои метки.
submeshes = []
omega = File(str(results / "omega.pvd"))
omega_boundary = File(str(results / "omega_boundary.pvd"))
omega_cells = File(str(results / "omega_cells.pvd"))
cell_file = File(str(results / "cells.pvd"))
boundary_file = File(str(results / "cell_boundaries.pvd"))
for j in range(1, n):
    for i in range(1, n):
        ax, ay = i * h, j * h
        patch_marks = MeshFunction("size_t", mesh, 2, 0)
        Rectangle(ax - h, ax + h, ay - h, ay + h).mark(patch_marks, 1)
        omega_i = SubMesh(mesh, patch_marks, 1)

        # Граница всей подобласти имеет метку 1.
        patch_boundary = MeshFunction("size_t", omega_i, 1, 0)
        patch_boundary.rename("omega_boundary", "Граница подобласти")
        Boundary().mark(patch_boundary, 1)
        # Один PVD: очередная подобласть — следующий шаг времени.
        omega << omega_i
        omega_boundary << patch_boundary

        # Шаг 5. Внутри подобласти отмечаем четыре составные ячейки.
        # Порядок: нижняя левая, нижняя правая, верхняя левая, верхняя правая.
        cell_marks = MeshFunction("size_t", omega_i, 2, 0)
        cell_marks.rename("cell_id", "Номер составной ячейки")
        patch_cells = []
        for dy in range(2):
            for dx in range(2):
                number = 1 + dx + 2 * dy
                xmin, ymin = ax + (dx - 1) * h, ay + (dy - 1) * h
                Rectangle(xmin, xmin + h, ymin, ymin + h).mark(cell_marks, number)

        omega_cells << cell_marks
        for number in range(1, 5):
            cell_mesh = SubMesh(omega_i, cell_marks, number)

            # Шаг 6. Границы ячейки по рисунку задания.
            # Сначала вся граница = 3, затем две стороны через A = 1 и 2.
            boundaries = MeshFunction("size_t", cell_mesh, 1, 0)
            boundaries.rename("boundary_id", "Метка границы ячейки")
            Boundary().mark(boundaries, 3)
            VerticalBoundary(ax).mark(boundaries, 1)
            HorizontalBoundary(ay).mark(boundaries, 2)

            cell_file << cell_mesh
            boundary_file << boundaries
            patch_cells.append((cell_mesh, boundaries))

        submeshes.append((omega_i, patch_boundary, cell_marks, patch_cells))


# Шаг 7. Сохраняем рисунок сетки: мелкая — чёрная, грубая — синяя.
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
