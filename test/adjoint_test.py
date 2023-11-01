import dolfinx
import ufl
from dolfinx.mesh import CellType, create_rectangle
from dolfinx import fem, nls
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

# mesh and function space
mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [1, 1]], [128, 128], CellType.triangle)
V = fem.FunctionSpace(mesh, ("CG", 1))
u = fem.Function(V)
v = ufl.TestFunction(V)

# this works OK
k = fem.Function(V)
k.x.array[:] = 1.0

#k = fem.Constant(mesh, PETSc.ScalarType((1.0))) #<- what should this be?

dx = ufl.Measure("dx")
F = ufl.inner((k*u + 1.0)*ufl.grad(u), ufl.grad(v))*dx # heat equation

# boundary conditions
def left(x): return np.isclose(x[0], 0)
def right(x): return np.isclose(x[0], 1)

fdim = mesh.topology.dim - 1

u1 = fem.Function(V)
boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)
with u1.vector.localForm() as loc:
    loc.set(0)
bc1 = fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

u2 = fem.Function(V)
boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right)
with u2.vector.localForm() as loc:
    loc.set(1)
bc2 = fem.dirichletbc(u2, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

bcs = [bc1, bc2]

# solve the problem
problem = fem.petsc.NonlinearProblem(F, u, bcs)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.solve(u)
# print(u.vector.array)

# set an arbitrary functinal
J = ufl.dot(u, u)*dx

ufl.derivative(J, k)  # <- fails here with: Can only create arguments automatically for non-indexed coefficients

# partial derivative of J w.r.t. k
dJdk = fem.petsc.assemble_vector(fem.form(ufl.derivative(J, k)))
dJdk.assemble()

# partial derivative of R w.r.t. k
dRdk = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(F, k))))  # partial derivative
dRdk.assemble()

# reset the boundary condition
with u2.vector.localForm() as loc:
    loc.set(0.0)

def compute_dJdk_function(value, dJdk=dJdk):
    # set k
    k.x.array[:] = value

    dJdk = fem.petsc.assemble_vector(fem.form(ufl.derivative(J, k)))
    dJdk.assemble()

    dRdk = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(F, k))))
    dRdk.assemble()

    with u2.vector.localForm() as loc:
        loc.set(0.0)

    # solve adjoint vector
    lhs = ufl.adjoint(ufl.derivative(F, u))
    rhs = -ufl.derivative(J, u)
    problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs)
    lmbda = problem.solve()

    # calculate derivative
    dJdk += dRdk*lmbda.vector
    dJdk.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    s = MPI.COMM_WORLD.reduce(np.sum(dJdk.array), op=MPI.SUM)
    return dJdk.array, s

def compute_dJdk_constant(value, dJdk=dJdk):
    # solve the adjoint vector
    k.value = value
    lhs = ufl.adjoint(ufl.derivative(F, u))
    rhs = -ufl.derivative(J, u)
    problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs)
    lmbda = problem.solve()

    dJdk += dRdk*lmbda.vector
    dJdk.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return dJdk

arr, s = compute_dJdk_function(0.1)
print(s)
