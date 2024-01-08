from mpi4py import MPI
from dolfinx import geometry
from dolfinx import fem
import numpy as np
import ufl
from petsc4py import PETSc

import dolfinx.nls.petsc
import dolfinx.fem.petsc

def wrap_constant_controls(controls):
    vars = {}
    for control in controls:
        for i in range(len(control)):
            vars[control[i]] = ufl.variable(control[i])
    return vars

class UflObjective:
    def __init__(self, J, u, controls):
        self.J = J
        self.J_form = fem.form(J)
        self.u = u
        self.controls = controls

        self.rhs = -ufl.derivative(J, u)
        self.rhs_form = fem.form(self.rhs)
        self.b = dolfinx.fem.petsc.create_vector(self.rhs_form)

        self.vars = wrap_constant_controls(self.controls)
        self.J_var_form = ufl.replace(self.J, self.vars)

        self.dJdk_form = [fem.form(ufl.diff(self.J_var_form, v)) for v in self.vars.values()]

    def assemble_adjoint_rhs_vector(self):
        with self.b.localForm() as b_loc:
            b_loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.b, self.rhs_form)
        self.b.assemble()
        return self.b
    
    def eval_dJdk(self):
        dJdk = np.zeros(len(self.dJdk_form))
        for i in range(len(self.dJdk_form)):
            dJdk[i] = fem.assemble_scalar(self.dJdk_form[i])
            dJdk[i] = MPI.COMM_WORLD.allreduce(dJdk[i], op=MPI.SUM)
        return dJdk
    
    def evaluate(self):
        J_value = fem.assemble_scalar(self.J_form)
        J_value = MPI.COMM_WORLD.allreduce(J_value, op=MPI.SUM)
        return J_value

class Point_wise_lsq_objective:
    def __init__(self, p_coords, f, controls, true_values = None):
        self.p_coords = p_coords
        self.f = f
        self.controls = controls
        self.b = f.copy()
        self.true_values = true_values

        self.vars = wrap_constant_controls(self.controls)

        self.evaluate_dofs_sensitivities()

    def evaluate_dofs_sensitivities(self):
        # sensitivity of values at p_coords w.r.t values at dofs
        domain = self.f.function_space.mesh
        points = np.array(self.p_coords)
        fdim = domain.topology.dim
        bb_tree = geometry.bb_tree(domain, fdim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

        # initialize result on each proc
        self.points = []
        self.cells = []
        self.dofs = []
        self.sensitivities = []
        self.true_values_map = []
        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                cell = colliding_cells.links(i)[0]
                dofs = self.f.function_space.dofmap.cell_dofs(cell)
                s = np.zeros(len(dofs))

                # there might exist nicer way but finite diff works too on (CG, 1)
                for j in range(len(dofs)):
                    val_0 = self.f.eval(point, cell)
                    self.f.x.array[dofs[j]] += 1.0
                    val_1 = self.f.eval(point, cell)
                    self.f.x.array[dofs[j]] -= 1.0
                    s[j] = val_1[0] - val_0[0]
            
                self.points.append(point)
                self.cells.append(cell)
                self.dofs.append(dofs)
                self.sensitivities.append(s)
                self.true_values_map.append(i)
    
    def assemble_adjoint_rhs_vector(self):
        with self.b.vector.localForm() as b_loc:
            b_loc.set(0)
        for i in range(len(self.points)):
            # contribution to dJdu of single point
            value = -2*(self.f.eval(self.points[i], self.cells[i])[0]-self.true_values[self.true_values_map[i]])
            self.b.x.array[self.dofs[i]] += self.sensitivities[i]*value
        return self.b.vector

    def eval_points(self):
        values = []
        for i in range(len(self.points)):
            values.append(self.f.eval(self.points[i], self.cells[i])[0])
        return values
    
    def eval_dJdk(self):
        dJdk = np.zeros(len(self.vars.keys()))
        return dJdk
    
    def evaluate(self):
        J_value = 0.0
        for i in range(len(self.points)):
            J_value += (self.f.eval(self.points[i], self.cells[i])[0]-self.true_values[self.true_values_map[i]])**2
        J_value = MPI.COMM_WORLD.allreduce(J_value, op=MPI.SUM)
        return J_value

class AdjointDerivative:
    def __init__(self, J, controls, form, forward_solver, u, bcs=None):
        self.form = form
        self.controls = controls
        self.J = J
        self.forward_solver = forward_solver
        self.u = u
        self.lmbda = u.copy()
        self.bcs = bcs or []

        self.vars = wrap_constant_controls(self.controls)
        self.var_form = ufl.replace(self.form, self.vars)

        self.dFdk_form = [fem.form(ufl.diff(self.var_form, var)) for var in self.vars.values()]
        self.dFdk = [fem.petsc.create_vector(form) for form in self.dFdk_form]

        # adjoint problem solver definition
        self.lhs = ufl.adjoint(ufl.derivative(self.form, u))
        self.lhs_form = fem.form(self.lhs)
        self.A = dolfinx.fem.petsc.create_matrix(self.lhs_form)

        self.adjoint_solver = PETSc.KSP().create(u.function_space.mesh.comm)
        self.adjoint_solver.setOperators(self.A)
        self.adjoint_solver.setType(PETSc.KSP.Type.PREONLY)
        self.adjoint_solver.setTolerances(atol=1e-16)
        self.adjoint_solver.getPC().setType(PETSc.PC.Type.LU)

    def solve_adjoint_problem(self):

        # lhs assembly
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_mat(self.A, self.lhs_form, bcs=self.bcs)
        self.A.assemble()

        # rhs assembly
        b = self.J.assemble_adjoint_rhs_vector()
        dolfinx.fem.petsc.apply_lifting(b, [self.lhs_form], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self.bcs)

        self.adjoint_solver.solve(b, self.lmbda.vector)
        self.lmbda.x.scatter_forward()
        return self.lmbda

    def forward(self, *args, **kwargs):
        self.forward_solver(*args, **kwargs)
    
    def compute_loss(self):
        return self.J.evaluate()

    def compute_gradient(self):

        # solve adjoint vector
        self.solve_adjoint_problem()

        #initialize empty gradient vector
        dJdk = self.J.eval_dJdk()
        for i in range(len(self.dFdk)):

            # reassemble adjoint stuff
            with self.dFdk[i].localForm() as dfdk_loc:
                dfdk_loc.set(0)
            dolfinx.fem.petsc.assemble_vector(self.dFdk[i], self.dFdk_form[i])
            self.dFdk[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # add dFdk*lmbda
            dJdk[i] += self.dFdk[i].dot(self.lmbda.vector)
            
        return dJdk
    
def taylor_test(loss, grad, k0, p=1e-3, n=5):
        g0 = grad(k0)
        l0 = loss(k0)
        reminder = []
        perturbance = []
        for i in range(0, n):
            l1 = loss(k0+p)
            reminder.append(l1 - l0 - np.sum(g0*p))
            perturbance.append(p)
            p /= 2
        conv_rate = convergence_rates(reminder, perturbance)
        return reminder, perturbance, conv_rate

def convergence_rates(r, p):
    cr = []
    for i in range(1, len(p)):
        cr.append(np.log(r[i] / r[i - 1])
                 / np.log(p[i] / p[i - 1]))
    return cr