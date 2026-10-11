from dolfin import *

mesh = Mesh("mesh.xml")
domain = MeshFunction("size_t", mesh, mesh.topology().dim()-1)


class TopBoundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > 1 - DOLFIN_EPS and x[0] <= 1.0 + DOLFIN_EPS
    
class LeftBoundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

class BottomBoundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < DOLFIN_EPS and x[0] <= 1.0 + DOLFIN_EPS

class RightBoundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] <= 1.0 + DOLFIN_EPS and x[1] >= 1.0 - DOLFIN_EPS



boundaries1 = TopBoundary1()
boundaries2 = LeftBoundary1()
boundaries3 = BottomBoundary1()
boundaries4 = RightBoundary1()


class TopBoundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > 1 - DOLFIN_EPS or x[0] >= 1.0 - DOLFIN_EPS or x[0] <= 1.5 - DOLFIN_EPS 
    
class LeftBoundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] <= 1.0 + DOLFIN_EPS and x[1] >= 1.0 - DOLFIN_EPS
    
class BottomBoundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < DOLFIN_EPS and x[0] >= 1.0 - DOLFIN_EPS

class RightBoundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > 1.5 - DOLFIN_EPS 

boundaries5 = TopBoundary2()
boundaries6 = LeftBoundary2()
boundaries7 = BottomBoundary2()
boundaries8 = RightBoundary2()



class LeftDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 1.0 + DOLFIN_EPS 

class RightDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 1.0 - DOLFIN_EPS 

left1 = LeftDomain()
right2 = RightDomain()

submesh = SubMesh(mesh, left1)
submesh2 = SubMesh(mesh, right2)




boundaries_sub1 = Measure("ds", domain=mesh, subdomain_data=submesh)
boundaries_sub2 = Measure("ds", domain=mesh, subdomain_data=submesh2)

boundaries1.mark(boundaries_sub1, 1)
boundaries2.mark(boundaries_sub1, 2)
boundaries3.mark(boundaries_sub1, 3)
boundaries4.mark(boundaries_sub1, 4)
boundaries5.mark(boundaries_sub2, 1)
boundaries6.mark(boundaries_sub2, 2)
boundaries7.mark(boundaries_sub2, 3)
boundaries8.mark(boundaries_sub2, 4)



File("submesh.pvd") << submesh
File("submesh2.pvd") << submesh2