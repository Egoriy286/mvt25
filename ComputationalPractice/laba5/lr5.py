from dolfin import *
import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path('results').mkdir(exist_ok=True)
set_log_level(LogLevel.ERROR)

# Строим сетку Gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 0)
gmsh.open('mesh.geo')
n = int(gmsh.parser.getNumber('n')[0])
gmsh.model.mesh.generate(3)
gmsh.write('mesh.msh')
gmsh.finalize()
source = meshio.read('mesh.msh')
tetrahedra = source.get_cells_type('tetra')
meshio.write('mesh.xml', meshio.Mesh(source.points, [('tetra', tetrahedra)]),
             file_format='dolfin-xml')

mesh = Mesh('mesh.xml')

# Нумеруем кубические области по координатам.
domains = MeshFunction('size_t', mesh, 3, 0)
for cell in cells(mesh):
    x = cell.midpoint()
    i = int(n*x[0])
    j = int(n*x[1])
    k = int(n*x[2])
    domains[cell] = i + n*j + n*n*k

# File('results/blocks.pvd') << domains

# Сохраняем пары: подсетка и метки её границы.
submeshes = []
submesh_file =File(f'results/submeshes/submesh.pvd')
facets_file = File(f'results/submeshes/facet.pvd')
for number in range(n**3):
    submesh = SubMesh(mesh, domains, number)
    facets = MeshFunction('size_t', submesh, 2, 0)
    DomainBoundary().mark(facets, 1)
    submeshes.append((submesh, facets))
    submesh_file << submesh
    facets_file << facets

print('Подсеток:', len(submeshes))

# Визуализация наружных граней куба.
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
surface = BoundaryMesh(mesh, 'exterior')
x, y, z = surface.coordinates().T
ax.plot_trisurf(x, y, z, triangles=surface.cells(), color='white',
                edgecolor='black', linewidth=0.2, shade=False, zorder=1)
# Синие линии на трёх видимых гранях куба.
for v in np.linspace(0,1,n+1):
    ax.plot([0,1],[v,v],[1,1],color='royalblue',lw=1)
    ax.plot([v,v],[0,1],[1,1],color='royalblue',lw=1)
    ax.plot([0,1],[0,0],[v,v],color='royalblue',lw=1)
    ax.plot([v,v],[0,0],[0,1],color='royalblue',lw=1)
    ax.plot([1,1],[0,1],[v,v],color='royalblue',lw=1)
    ax.plot([1,1],[v,v],[0,1],color='royalblue',lw=1)
ax.set(xlabel='x',ylabel='y',zlabel='z',title='Сетка 3D')
ax.set_box_aspect([1,1,1])
plt.tight_layout()
plt.savefig('results/mesh.png', dpi=150)
plt.show()
