# dolfin-convert mesh.msh mesh.xml 
from dolfin import *

mesh = Mesh("mesh.xml")

class inner_circle(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) <= 0.2**2 + DOLFIN_EPS

class out_circle(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) >= 0.2**2 - DOLFIN_EPS

circle = inner_circle()
out_cir = out_circle()

submesh = SubMesh(mesh, circle)
submesh2 = SubMesh(mesh, out_cir)

File("submesh.pvd") << submesh
File("submesh2.pvd") << submesh2