from dolfin import *
import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Параметры сетки задаются в mesh.geo: 10 крупных блоков,
# по 4 мелких разбиения внутри блока по каждой координате.
Path('results/submeshes').mkdir(parents=True, exist_ok=True)
set_log_level(LogLevel.ERROR)

# Строим сетку Gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 0)
gmsh.open('mesh.geo')
gmsh.model.mesh.generate(2)
gmsh.write('mesh.msh')
gmsh.finalize()
source = meshio.read('mesh.msh')
quad = source.get_cells_type('quad')
# Recombine даёт четырёхугольники. Для DOLFIN SubMesh делим каждый на 2 треугольника.
triangles = np.vstack([quad[:, [0,1,2]], quad[:, [0,2,3]]])
meshio.write('mesh.xml', meshio.Mesh(source.points[:,:2], [('triangle', triangles)]),
             file_format='dolfin-xml')

mesh = Mesh('mesh.xml')

# Выделение областей только по координатам, без Physical Groups в Gmsh.
domains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
for cell in cells(mesh):
    x = cell.midpoint()
    i = int(10*x[0])
    j = int(10*x[1])
    domains[cell] = i + 10*j

File('results/blocks.pvd') << domains

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# В каждой подсетке отмечаем всю границу единицей.
summary = []
for number in range(100):
    submesh = SubMesh(mesh, domains, number)
    facets = MeshFunction('size_t', submesh, 1, 0)
    Boundary().mark(facets, 1)
    ds_sub = Measure('ds', domain=submesh, subdomain_data=facets)
    measure = assemble(Constant(1.)*dx(domain=submesh))
    boundary_measure = assemble(Constant(1.)*ds_sub(1))
    assert abs(measure - 0.01) < 1e-10
    assert abs(boundary_measure - 0.4) < 1e-10
    File(f'results/submeshes/submesh_{number:04d}.pvd') << submesh
    File(f'results/submeshes/facets_{number:04d}.pvd') << facets
    summary.append([number, submesh.num_cells(), measure, boundary_measure])

np.savetxt('results/submeshes.csv', summary, delimiter=',',
           header='number,elements,measure,boundary_measure', comments='')
print('Подсеток:', len(summary))
print('Элементов в подсетке:', summary[0][1])
print('Суммарная мера:', sum(row[2] for row in summary))
print('Границы всех подсеток имеют метку 1')

# Визуализация: чёрная мелкая сетка и синие линии крупных блоков.
fig, ax = plt.subplots(figsize=(6,6))
for cell in quad:
    xy = source.points[np.r_[cell,cell[0]]]
    ax.plot(xy[:,0], xy[:,1], color='black', lw=0.3)
for v in np.linspace(0,1,11):
    ax.axvline(v, color='royalblue', lw=1)
    ax.axhline(v, color='royalblue', lw=1)
ax.set(xlim=(0,1), ylim=(0,1), xlabel='x', ylabel='y', title='Transfinite 2D', aspect='equal')
plt.tight_layout()
plt.savefig('results/mesh.png', dpi=150)
plt.show()
