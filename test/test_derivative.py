from mpi4py import MPI
import dolfinx
import ufl
from petsc4py import PETSc
import numpy as np
import unittest

from heat_battery.optimization import (
    UflObjective, Point_wise_lsq_objective, AdjointDerivative, taylor_test)

import dolfinx.nls.petsc
import dolfinx.fem.petsc

class TestDerivative(unittest.TestCase):
    def setUp(self) -> None:
        # mesh and function spaces
        mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [1, 1]], [64, 64], dolfinx.mesh.CellType.triangle)
        V = dolfinx.fem.functionspace(mesh, ("P", 1)) # solution space
        self.u = dolfinx.fem.Function(V)  # solution trial
        v = ufl.TestFunction(V)  # solution test

        self.k = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1.0, 1.0)))

        self.dx = ufl.Measure("dx")
        self.F = ufl.inner(ufl.grad(self.u), ufl.grad(v))*self.dx # heat equation
        self.F += sum(self.k)*v*self.dx # add source of heat

        # boundary conditions
        def left(x): return np.isclose(x[0], 0)

        fdim = mesh.topology.dim - 1

        u1 = dolfinx.fem.Function(V)
        boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)
        with u1.x.petsc_vec.localForm() as loc:
            loc.set(0)
        bc1 = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

        self.bcs = [bc1]

        # solve the problem
        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1e-16,
            "snes_rtol": 1e-16,
            "ksp_rtol": 1e-16,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        }
        self.solver = dolfinx.fem.petsc.NonlinearProblem(
            self.F,
            self.u,
            bcs=self.bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="testderivative",
        )
        self.forward = lambda: self.solver.solve()
    
    def test_ufl_objective(self):
        J = ufl.inner(self.u, self.u)*self.dx

        controls = [self.k]
        Jr = UflObjective(J, self.u, controls)
        adjoint = AdjointDerivative(Jr, controls, self.F, self.forward, self.u, self.bcs)

        def grad(value):
            self.k.value[:] = value
            adjoint.forward()
            return adjoint.compute_gradient()

        def loss(value):
            self.k.value[:] = value
            self.solver.solve()
            l = Jr.evaluate()
            return l
        
        k0 = np.array([1, 1])
        r = taylor_test(loss, grad, k0)
        self.assertTrue(np.allclose(r[2], 2.0, atol=0.1), f"Taylor test failed - Convergence rate: {r[2]}")

    def test_lsq_objective(self):
        points = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
        true_values = [0.0, 0.0]
        controls = [self.k]
        Jr = Point_wise_lsq_objective(points, self.u, controls, true_values)
        adjoint = AdjointDerivative(Jr, controls, self.F, self.forward, self.u, self.bcs)

        def grad(value):
            self.k.value[:] = value
            adjoint.forward()
            return adjoint.compute_gradient()

        def loss(value):
            self.k.value[:] = value
            self.solver.solve()
            l = Jr.evaluate()
            return l
        
        k0 = np.array([1, 1])
        r = taylor_test(loss, grad, k0)
        self.assertTrue(np.allclose(r[2], 2.0, atol=0.1), f"Taylor test failed - Convergence rate: {r[2]}")

    def test_optimisation(self):
        points = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
        true_values = [0.0, 0.0]
        controls = [self.k]
        Jr = Point_wise_lsq_objective(points, self.u, controls, true_values)
        adjoint = AdjointDerivative(Jr, controls, self.F, self.forward, self.u, self.bcs)

        # find zero
        k_value = np.array([1.0, 1.0])
        alpha = 0.5
        # Vanila gradient descent
        for i in range(150):
            adjoint.forward()
            g = adjoint.compute_gradient()
            self.k.value += -alpha*g
            #if MPI.COMM_WORLD.rank == 0:
            #    print(l, self.k.value)

        self.assertTrue(np.allclose(self.k.value, 0.0, atol=1e-6), f"k_value should be [0.0, 0.0] not {self.k.value}")

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()