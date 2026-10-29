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
gmsh.model.mesh.generate(3)
gmsh.write('mesh.msh')
gmsh.finalize()
source = meshio.read('mesh.msh')
hexa = source.get_cells_type('hexahedron')
# Приводим вершины кубиков к одному порядку по координатам.
# Тогда при разбиении на тетраэдры диагонали общих граней совпадают.
xyz = source.points[hexa]
center = xyz.mean(axis=1)
code = ((xyz[:,:,0] > center[:,None,0]).astype(int)
        + 2*(xyz[:,:,1] > center[:,None,1])
        + 4*(xyz[:,:,2] > center[:,None,2]))
hexa = np.take_along_axis(hexa, np.argsort(code, axis=1), axis=1)
hexa = hexa[:, [0,1,3,2,4,5,7,6]]
# Gmsh-сетку из гексаэдров сохраняем; для DOLFIN каждый кубик делим на 6 тетраэдров.
tetrahedra = np.vstack([hexa[:, face] for face in
    [[0,1,2,6], [0,2,3,6], [0,3,7,6], [0,7,4,6], [0,4,5,6], [0,5,1,6]]])
meshio.write('mesh.xml', meshio.Mesh(source.points, [('tetra', tetrahedra)]),
             file_format='dolfin-xml')

mesh = Mesh('mesh.xml')

# Выделение областей только по координатам, без Physical Groups в Gmsh.
domains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
for cell in cells(mesh):
    x = cell.midpoint()
    i = int(10*x[0])
    j = int(10*x[1])
    k = int(10*x[2])
    domains[cell] = i + 10*j + 100*k

File('results/blocks.pvd') << domains

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# В каждой подсетке отмечаем всю границу единицей.
summary = []
for number in range(1000):
    submesh = SubMesh(mesh, domains, number)
    facets = MeshFunction('size_t', submesh, 2, 0)
    Boundary().mark(facets, 1)
    ds_sub = Measure('ds', domain=submesh, subdomain_data=facets)
    measure = assemble(Constant(1.)*dx(domain=submesh))
    boundary_measure = assemble(Constant(1.)*ds_sub(1))
    assert abs(measure - 0.001) < 1e-10
    assert abs(boundary_measure - 0.06) < 1e-10
    File(f'results/submeshes/submesh_{number:04d}.pvd') << submesh
    File(f'results/submeshes/facets_{number:04d}.pvd') << facets
    summary.append([number, submesh.num_cells(), measure, boundary_measure])

np.savetxt('results/submeshes.csv', summary, delimiter=',',
           header='number,elements,measure,boundary_measure', comments='')
print('Подсеток:', len(summary))
print('Элементов в подсетке:', summary[0][1])
print('Суммарная мера:', sum(row[2] for row in summary))
print('Границы всех подсеток имеют метку 1')

# Визуализация наружных граней куба.
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
for v in np.linspace(0,1,41):
    color = 'royalblue' if np.isclose(v*10, round(v*10)) else 'black'
    width = 1 if color == 'royalblue' else 0.25
    for side in [0.,1.]:
        ax.plot([0,1],[v,v],[side,side],color=color,lw=width)
        ax.plot([v,v],[0,1],[side,side],color=color,lw=width)
        ax.plot([0,1],[side,side],[v,v],color=color,lw=width)
        ax.plot([v,v],[side,side],[0,1],color=color,lw=width)
        ax.plot([side,side],[0,1],[v,v],color=color,lw=width)
        ax.plot([side,side],[v,v],[0,1],color=color,lw=width)
ax.set(xlabel='x',ylabel='y',zlabel='z',title='Transfinite 3D')
ax.set_box_aspect([1,1,1])
plt.tight_layout()
plt.savefig('results/mesh.png', dpi=150)
plt.show()
