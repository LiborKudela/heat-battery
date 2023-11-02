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

# let's say I have bunch of these and I want to get grad J w.r.t them
#k = fem.Constant(mesh, PETSc.ScalarType((1.0))) #<- what should this be?

dx = ufl.Measure("dx")
F = ufl.inner(ufl.grad(u), ufl.grad(v))*dx # heat equation
F += k*v*dx # add source of heat

# boundary conditions
def left(x): return np.isclose(x[0], 0)
def right(x): return np.isclose(x[0], 1)

fdim = mesh.topology.dim - 1

u1 = fem.Function(V)
boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)
with u1.vector.localForm() as loc:
    loc.set(0)
bc1 = fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

bcs = [bc1]

# solve the problem
problem = fem.petsc.NonlinearProblem(F, u, bcs)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

# how big is the solution?
J = ufl.dot(u, u)*dx
J_form = fem.form(J)

# partial derivative of J w.r.t. k
dJdk_form = fem.form(ufl.derivative(J, k))

# partial derivative of R w.r.t. k
dRdk_form = fem.form(ufl.adjoint(ufl.derivative(F, k)))

lhs = ufl.adjoint(ufl.derivative(F, u))
rhs = -ufl.derivative(J, u)
problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs)

def compute_dJdk_where_k_is_function(value):
    # set k
    k.x.array[:] = value

    # solve new solution vector
    solver.solve(u)

    # reassemble adjoint stuff
    dJdk = fem.petsc.assemble_vector(dJdk_form)
    dJdk.assemble()
    dRdk = fem.petsc.assemble_matrix(dRdk_form)
    dRdk.assemble()

    # solve adjoint vector
    lmbda = problem.solve()

    # calculate derivative (GRADIENT w.r.t. dofs of k)
    dJdk += dRdk*lmbda.vector
    dJdk.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # calculate size
    J_value = fem.assemble_scalar(J_form)
    J_value = MPI.COMM_WORLD.allreduce(J_value, op=MPI.SUM) # J value

    s = np.sum(dJdk.array)
    s = MPI.COMM_WORLD.allreduce(s, op=MPI.SUM) # grad when k constant
    return s, J_value

# find zero
k_value = 10.0
alpha = 0.5
# Vanila gradient descent
for i in range(100):
    s, J_value = compute_dJdk_where_k_is_function(k_value)
    if MPI.COMM_WORLD.rank == 0:
        print(s, J_value, k_value)
    k_value += -alpha*s



