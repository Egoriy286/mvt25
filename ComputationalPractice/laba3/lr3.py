from dolfin import *

mesh = Mesh("mesh.xml")

class LeftDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 1.0 + DOLFIN_EPS


class RightDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 1.0 - DOLFIN_EPS


left_domain = LeftDomain()
right_domain = RightDomain()


submesh_left = SubMesh(mesh, left_domain)
submesh_right = SubMesh(mesh, right_domain)


# Сохраняем SubMesh
File("submesh_left.pvd") << submesh_left
File("submesh_right.pvd") << submesh_right

class TopZLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 1.0) and x[0] <= 1.0 + DOLFIN_EPS
    
class BottomZLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0.0) and x[0] <= 1.0 + DOLFIN_EPS


class TopYLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0) and x[0] <= 1.0 + DOLFIN_EPS


class BottomYLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0) and x[0] <= 1.0 + DOLFIN_EPS


class TopXLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)

class BottomXLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
    
class TopZRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 1.0) and x[0] >= 1.0 - DOLFIN_EPS
    
class BottomZRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0.0) and x[0] >= 1.0 - DOLFIN_EPS

class TopYRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0) and x[0] >= 1.0 - DOLFIN_EPS


class BottomYRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0) and x[0] >= 1.0 - DOLFIN_EPS


class TopXRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.5)

class BottomXRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)


# Mark facets левой подобласти
facets_left = MeshFunction(
    "size_t",
    submesh_left,
    submesh_left.topology().dim() - 1,
    0
)

# Левая область
TopZLeftBoundary().mark(facets_left, 1)
BottomZLeftBoundary().mark(facets_left, 2)
TopYLeftBoundary().mark(facets_left, 3)
BottomYLeftBoundary().mark(facets_left, 4)
TopXLeftBoundary().mark(facets_left, 5)
BottomXLeftBoundary().mark(facets_left, 6)


ds_left = Measure(
    "ds",
    domain=submesh_left,
    subdomain_data=facets_left
)


# Mark facets правой подобласти
facets_right = MeshFunction(
    "size_t",
    submesh_right,
    submesh_right.topology().dim() - 1,
    0
)

# Правая область
TopZRightBoundary().mark(facets_right, 1)
BottomZRightBoundary().mark(facets_right, 2)
TopYRightBoundary().mark(facets_right, 3)
BottomYRightBoundary().mark(facets_right, 4)
TopXRightBoundary().mark(facets_right, 5)
BottomXRightBoundary().mark(facets_right, 6)


# Measure для правой подобласти
ds_right = Measure(
    "ds",
    domain=submesh_right,
    subdomain_data=facets_right
)

File("facets_left.pvd") << facets_left
File("facets_right.pvd") << facets_right

