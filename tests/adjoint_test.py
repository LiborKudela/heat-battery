import dolfinx
import ufl
from dolfinx.mesh import CellType, create_rectangle
from dolfinx import fem, nls
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

# mesh and function space
mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [1, 1]], [3, 3], CellType.triangle)
V = fem.FunctionSpace(mesh, ("CG", 1))
u = fem.Function(V)
v = ufl.TestFunction(V)

# this works OK
#mu = fem.Function(V)
#mu.x.array[:] = 1.0

# but ow to define this so I can have multiple constants (lets say coefficients of polynomial material property)
# in th form and get sensitivity of some a functional J w.r.t these constants (or at least single one)

mu = fem.Constant(mesh, PETSc.ScalarType((1.0))) #<- what should this be?
#ERROR:UFL:Can only create arguments automatically for non-indexed coefficients.

dx = ufl.Measure("dx")
F = ufl.inner(mu*ufl.grad(u), ufl.grad(v))*dx # heat equation

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
J = mu*ufl.dot(u, u)*dx

ufl.derivative(J, mu)  # <- fails here with: Can only create arguments automatically for non-indexed coefficients

# partial derivative of J w.r.t. mu
dJdmu = fem.petsc.assemble_vector(fem.form(ufl.derivative(J, mu)))
dJdmu.assemble()

# partial derivative of R w.r.t. mu
dRdmu = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(F, mu))))  # partial derivative
dRdmu.assemble()

# reset the boundary condition
with u2.vector.localForm() as loc:
    loc.set(0.0)

# solve the adjoint vector
lhs = ufl.adjoint(ufl.derivative(F, u))
rhs = -ufl.derivative(J, u)
problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs)
lmbda = problem.solve()

dJdmu += dRdmu*lmbda.vector
dJdmu.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
print(dJdmu.array)