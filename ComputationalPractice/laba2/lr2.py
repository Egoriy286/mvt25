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

class LeftTopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)


class LeftLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)


class LeftBottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)


class LeftRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)


# Mark facets левой подобласти
facets_left = MeshFunction(
    "size_t",
    submesh_left,
    submesh_left.topology().dim() - 1,
    0
)

LeftTopBoundary().mark(facets_left, 1)
LeftLeftBoundary().mark(facets_left, 2)
LeftBottomBoundary().mark(facets_left, 3)
LeftRightBoundary().mark(facets_left, 4)

ds_left = Measure(
    "ds",
    domain=submesh_left,
    subdomain_data=facets_left
)

class RightTopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)

class RightLeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)


class RightBottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)


class RightRightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.5)

# Mark facets правой подобласти
facets_right = MeshFunction(
    "size_t",
    submesh_right,
    submesh_right.topology().dim() - 1,
    0
)

RightTopBoundary().mark(facets_right, 1)
RightLeftBoundary().mark(facets_right, 2)
RightBottomBoundary().mark(facets_right, 3)
RightRightBoundary().mark(facets_right, 4)


# Measure для правой подобласти
ds_right = Measure(
    "ds",
    domain=submesh_right,
    subdomain_data=facets_right
)

File("facets_left.pvd") << facets_left
File("facets_right.pvd") << facets_right

