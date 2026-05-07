from typing import List
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from mpi4py import MPI
import numpy as np
from ..simulations import Simulation
from .derivatives import AdjointDerivative, Point_wise_lsq_objective, ForwardDerivative_dudk
from scipy.linalg import block_diag


@dataclass(frozen=True)
class JacobianRankAssessment:
    """Result of :meth:`SteadyStateComparer.get_optimization_problem_posedness`."""

    singular_values: np.ndarray
    jacobian_shape: tuple[int, int]
    effective_rank_absolute: int
    effective_rank_relative: int
    sigma_lim: float

class SteadyStateComparer:
    def __init__(self, sim: Simulation, inputs, outputs):
        self.sim = sim
        assert len(inputs) == len(outputs), "not the same length"
        self.n = len(inputs)
        self.inputs: List[dict] = inputs
        self.outputs: List[np.ndarray] = outputs
        
        self.data = [pd.Series()]*self.n
        self.domain_data = [None]*self.n 

    def get_k(self, m=None):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)
        k = []
        for i in m:
            k.append(self.sim.mats[i].k.get_values())
        return np.concatenate(k)
    
    def set_k(self, k, m=None):
        self.sim.mats.set_property_values(k, 'k', m)

    def generate_loss_gradient_for_material(self, m=None, batch_size=None):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)

        controls = [self.sim.mats[i].k.fem_const for i in m]
        transform_jac = block_diag(*[self.sim.mats[i].k.transform_jac for i in m])
        points = self.sim.T_probes_coords
        true_vals = None
       
        J = Point_wise_lsq_objective(points, self.sim.T, controls, true_vals)
        adjoint_derivative = AdjointDerivative(J, controls, self.sim.Fss, self.sim.solve_steady, self.sim.T)

        def gradient(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()

            self.set_k(k, m=m)
            g = np.zeros_like(original_k)
            l = 0.0

            # select batch
            if batch_size is not None:
                idx_batch = np.random.choice(self.n, size=batch_size, replace=False)
                idx_batch = MPI.COMM_WORLD.bcast(idx_batch)
            else:
                idx_batch = np.arange(self.n)

            try:
                for i in idx_batch:
                    input_data = self.inputs[i]
                    output_data = self.outputs[i]
                    J.true_values = output_data
                    adjoint_derivative.forward(**input_data)
                    _g = adjoint_derivative.compute_gradient()
                    g += _g.dot(transform_jac)
                    l += adjoint_derivative.compute_loss()
            except Exception as e:
                raise e
            finally:
                self.set_k(original_k, m=m)
                self.sim.T.x.array[:] = original_T
            return g, l

        return gradient
    
    def generate_solution_jacobian(self, m=None, full_domain=False):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)

        controls = [self.sim.mats[i].k.fem_const for i in m]
        transform_jac = block_diag(*[self.sim.mats[i].k.transform_jac for i in m])
        points = self.sim.T_probes_coords if not full_domain else None
        n_points = len(points) if not full_domain else self.sim.T.x.array.shape[0]

        forward_jacobian = ForwardDerivative_dudk(controls, self.sim.Fss, self.sim.solve_steady, self.sim.T, p_coords=points)
        
        def jacobian(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()

            self.set_k(k, m=m)
            j = np.zeros((len(k), n_points*self.n))

            for i in range(self.n):
                input_data = self.inputs[i]
                forward_jacobian.forward(**input_data)
                _j = forward_jacobian.compute_jacobian()
                z = i*_j.shape[1]
                j[:, z:z+_j.shape[1]] = transform_jac.T.dot(_j)
                
            self.set_k(original_k, m=m)
            self.sim.T.x.array[:] = original_T

            return j
        return jacobian

    def get_optimization_problem_posedness(
        self,
        k=None,
        m=None,
        full_domain=False,
        sigma_lim=1e-15,
    ):
        """Screen whether the current experiment set is informative enough to fit ``k``.

        Use this **before or alongside any optimisation** that adjusts ``k``
        against these steady experiments—gradient methods (e.g. Adam, BFGS).
        Small singular values or low effective rank relative to the number of ``k`` 
        coefficients spell trouble for **all** such algorithms (slow progress, 
        instability, dependence on step size / regularisation).

        It judges whether stacked steady scenarios (boundary inputs + measured
        probe temperatures in ``inputs`` / ``outputs``) excite the conductivity
        degrees of freedom sufficiently.

        The method builds the sensitivity matrix
        ``∂(model probe temperatures) / ∂k`` — same layout as
        :meth:`generate_solution_jacobian` — at the linearisation point ``k``, runs
        ``numpy.linalg.svd(..., compute_uv=False)``, and summarises:

        * **singular_values** — strength of each identifiable direction in
          parameter space. A long tail near zero means some combinations of
          ``k`` barely move the predictions: the experiment design is **poor**
          or redundant for those modes. Some experiments or sensor placement
          do not bring enough new information to the problem (redundancy).
        * **effective_rank_absolute** / **effective_rank_relative** — counts of
          singular values above ``sigma_lim`` and above ``sigma_lim * σ_max``.
          Always ``effective_rank_* ≤ min(n_k, n_o) ≤ n_k``, where ``n_k`` is
          ``jacobian_shape[0]`` (fitted ``k`` coefficients) and ``n_o`` is
          ``jacobian_shape[1]`` (stacked model probes × scenarios): the rank of
          ``J`` cannot exceed the smaller matrix dimension. Compare
          ``effective_rank_*`` to ``n_k``: if it is much smaller, the fit is
          **ill-posed** unless you reduce parameters, add richer experiments
          (more distinct steady boundary conditions, e.g. heating powers or
          ambient temperatures), or regularise.

        This does **not** replace statistical noise analysis; it only reflects
        **deterministic sensitivity** of the forward model at ``k``.

        Parameters
        ----------
        k
            Linearisation point for ``k`` (concatenated material coefficients).
            Uses :meth:`get_k` when omitted (typically current trial ``k``).
        m
            Subset of material indices whose ``k`` rows appear in the Jacobian
            (passed through to :meth:`generate_solution_jacobian`).
        full_domain
            Same as :meth:`generate_solution_jacobian` (probe rows vs full DOFs).
        sigma_lim
            Absolute cutoff used for ``effective_rank_absolute``; also scaled by
            ``σ_max`` for ``effective_rank_relative``.

        Returns
        -------
        :class:`JacobianRankAssessment`
            Object containing the singular values, Jacobian shape, effective rank, and sigma_lim.
        """
        if k is None:
            k = self.get_k(m=m)
        jac_fn = self.generate_solution_jacobian(m=m, full_domain=full_domain)
        J = jac_fn(np.asarray(k))
        s = np.linalg.svd(J, compute_uv=False)
        smax = float(s[0]) if s.size > 0 else 0.0
        rank_abs = int(np.sum(s >= sigma_lim))
        rank_rel = int(np.sum(s >= sigma_lim * smax)) if smax > 0 else 0
        return JacobianRankAssessment(
            singular_values=s,
            jacobian_shape=(J.shape[0], J.shape[1]),
            effective_rank_absolute=rank_abs,
            effective_rank_relative=rank_rel,
            sigma_lim=float(sigma_lim),
        )

    def generate_loss_component_jacobian(self, m=None):
        return self.generate_solution_jacobian(m=m, full_domain=False)
    
    def generate_loss_for_material(self, m):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)

        points = self.sim.T_probes_coords
        controls = [self.sim.mats[i].k.fem_const for i in m]
        true_vals = None
        J = Point_wise_lsq_objective(points, self.sim.T, controls, true_vals)

        def loss_function(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()
            self.set_k(k, m=m)

            l = 0.0
            for i in range(self.n):
                input_data = self.inputs[i]
                output_data = self.outputs[i]
                J.true_values = output_data
                self.sim.solve_steady(**input_data)
                l += J.evaluate()

            self.set_k(original_k, m=m)
            self.sim.T.x.array[:] = original_T
            return l
        return loss_function
    
    def generate_getter_for_material(self, m):
        def restricted_getter():
            return self.get_k(m=m)
        return restricted_getter
    
    def generate_setter_for_material(self, m):
        def restricted_setter(k):
            self.set_k(k, m=m)
        return restricted_setter

    def update(self):
        for i, input_data in enumerate(self.inputs):
            probes = self.sim.solve_steady(**input_data)
            self.data[i] = probes.get_value('T')
            self.domain_data[i] = self.sim.T.copy()
        
    def compare_plot(self):
        if MPI.COMM_WORLD.rank == 0:
            figs = []
            for i in range(self.n):
                fig = go.Figure()
                fig.add_bar(y=self.data[i], name='Simulation')
                fig.add_bar(y=self.outputs[i], name='Experiment')
                figs.append(fig)
                #TODO: add titles to compare charts
                #TODO: add x index so there are names
            return figs
        
    def material_plot(self, m=None, property='k', include_density=False):
        return self.sim.material_plot(m=m, property=property, include_density=include_density, T=self.domain_data)
    
    def domain_state_plot(self):
        if self.domain_data[0] is not None:
            return self.sim.domain_state_plot(T=self.domain_data)
        
    def print_data(self):
        if MPI.COMM_WORLD.rank == 0:
            print(self.data)