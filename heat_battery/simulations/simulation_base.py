"""Finite-element heat conduction simulations on DOLFINx (FEniCSx).

This module defines :class:`Simulation`, the base class used to build transient
and steady thermal models from Gmsh-generated meshes (``.msh``/``.ad``), material
metadata, and weak-form :mod:`~heat_battery.simulations.terms`.

Typical usage is to subclass ``Simulation``, override hooks such as
``define_form_subdomain_terms`` and probe setup, then call
``solve_steady`` or ``solve_unsteady``. The unsteady driver performs adaptive
time stepping, couples optional :class:`~heat_battery.simulations.terms_base.Term`
instances, and streams probe data via :class:`~heat_battery.simulations.probing.Probe_writer`
to CSV files, in-memory buffers, or a PostgreSQL-backed project database.

**Outputs**

* **Probes** — scalar time series (temperatures, fluxes, custom quantities)
  configured on the steady/unsteady probe writers.
* **XDMF** — optional full-domain temperature field history when ``xdmf_file``
  and ``result_dir`` are set in :meth:`Simulation.solve_unsteady`.

**Checkpoints**

:meth:`Simulation.solve_unsteady` supports periodic checkpoints when
``checkpoint_dir`` and ``checkpoint_dt`` are given. State is written with
adios4dolfinx; probe destinations participate via their own checkpoint hooks
(e.g. CSV truncation to a saved byte length). When XDMF output is enabled, a
copy of the ``.xdmf``/``.h5`` pair is also stored under the checkpoint directory
so a resume can roll domain output back in sync with the simulation state.

**MPI**

The code assumes a distributed mesh and uses ``MPI.COMM_WORLD`` for barriers and
collective file semantics; user-facing prints should go through
:meth:`Simulation.print_r0` where only rank 0 should speak.
"""

from mpi4py import MPI
import pandas as pd
from petsc4py import PETSc
import numpy as np
import ufl
import time
import os
import pyvista
from dolfinx import io, fem, __version__, plot
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import plotly.graph_objects as go
import plotly.express as px
from ..config import get_config_item
import adios4dolfinx
from pathlib import Path
import datetime
from typing import Callable, Any
from .probing import Probe_writer
from ..materials import MaterialsSet
from ..utilities import save_data_binary, load_data_binary, ProgressBar, save_data_json, load_data_json
import shutil

class Simulation():
    """Base class for mesh-backed thermal FEM models and time stepping.

    Constructs solvers, probes, and forms from ``geometry_dir`` and ``model_name``.
    Subclasses add boundary/source terms and application-specific behaviour.
    """

    def __init__(self,
        geometry_dir='meshes/experiment', 
        model_name='mesh_2d',
        build_solvers=[
            'derivative',
            'steady',
            'unsteady'],
        ):
        """
        Simulation class is used to build numerical finite element models.
        Usualy it is not used directly, but rather through derived classes, 
        which will have more specific functionality such as probes, external 
        bondaries and complex Terms definitions. For more details on how to 
        derive from this class see examples.

        Args:
            geometry_dir: Directory to load mesh data from
            result_dir: Directory to save simulation results to
            result_database: Name of the database to save results to
            model_name: Name of the mesh files to load (without file extension)
            build_solvers: List of solvers to build
        """
 
        self.geometry_dir = geometry_dir
        self.model_name = model_name

        self.print_r0(f"Dolfinx version: {__version__}")
        self.load_geometry()
        self.define_measures()
        self.create_function_spaces()
        self.create_functions()
        self.calculate_volumes_of_subdomains()
        self.calculate_areas_of_surfaces()
        self.create_form_constants()
        self.create_form_terms_presized_lists()
        self.define_form_subdomain_terms()
        self.resolve_form_terms_update_callbacks()
        self.resolve_form_terms_next_step_callbacks()
        self.resolve_form_terms_converged_callbacks()
        self.resolve_form_terms_adaptation_callbacks()

        for s_type in build_solvers:
            if s_type == 'derivative':
                pass
                #self.create_time_derivative_form_solver()
            elif s_type == 'steady':
                self.create_steady_state_form_solver()
                self.create_steady_probe_writer()
            elif s_type == 'unsteady':
                self.create_unsteady_form_solver()
                self.create_unsteady_probe_writer()
            else:
                print(f"Uknown solver : {s_type}")
        
        self.create_forms_for_calculating_temperature_spectrum()
        self.create_static_vtk_data()

    def print_r0(self, *args, **kwargs):
        """Print only on MPI rank 0 (avoids duplicate console output)."""
        if MPI.COMM_WORLD.rank == 0:
            print(*args, **kwargs)

    def load_geometry(self):
        """Load mesh and metadata from ``geometry_dir`` / ``model_name``.

        Populates ``domain``, ``cell_tags``, ``facet_tags``, ``mats``,
        boundary name maps, and spatial coordinate ``x`` with Jacobian
        scaling ``jac`` from the ``.msh`` / ``.ad`` pair.
        """
        self.geo_path = os.path.join(self.geometry_dir, self.model_name)

        # load metadata
        self.geo_meta = load_data_binary(f'{self.geo_path}.ad')

        # load domain
        self.dim = self.geo_meta['dim']
        mesh_data = io.gmsh.read_from_msh(f'{self.geo_path}.msh', MPI.COMM_WORLD, 0, gdim=self.dim)
        self.domain = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        self.facet_tags = mesh_data.facet_tags
        # instantiate materials
        self.mats = MaterialsSet(self.domain, self.geo_meta['materials'])
        self.subdomain_map = self.mats.key_map

        # boundary surface names defined in geometry metadata
        self.bcs = self.geo_meta['boundaries']
        self.bcs_map = {name: i for i, name in enumerate(self.bcs.keys())}

        # define spatial coordinate for weak form evaluation and jacobian expression
        self.x = ufl.SpatialCoordinate(self.domain)
        self.jac = self.geo_meta['jac_f'](self.x)

    def define_measures(self):
        """Create volumetric ``dx`` and surface ``ds`` / ``dS`` UFL measures on tagged facets."""
        self.dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.dS = ufl.Measure("dS", domain=self.domain, subdomain_data=self.facet_tags)

    def create_function_spaces(self):
        """Allocate the scalar Lagrange P1 space for temperature."""
        #self.initial_condition = lambda x: np.full((x.shape[1],), self.T0)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))

    def create_functions(self):
        """Create temperature fields ``T``, ``T_n``, ``dT``, and interpolation helper ``T_temp``."""
        # temperature in current time step
        self.T = fem.Function(self.V)
        self.T_v = ufl.TestFunction(self.V)
        self.T.name = "T"
        self.T.x.array[:] = 20.0

        # temperature in previous time step
        self.T_n = fem.Function(self.V)
        self.T_n.name = "T_n"
        self.T_n.x.array[:] = 20.0

        # temperature time dirivative at state T_n and previous time t_n
        self.dT = fem.Function(self.V)
        self.dT.name = "dT"
        self.dT.x.array[:] = 0.0

        # temperature temporary handlign
        self.T_temp = fem.Function(self.V)
        self.T_temp.name = "T"
        self.T_temp.x.array[:] = 20.0

    def calculate_volumes_of_subdomains(self):
        """Fill ``V_subdomain`` with integrated cell volumes per material tag (MPI-reduced)."""
        self.V_subdomain = []
        for i, mat in enumerate(self.mats, 1):
            V = fem.assemble_scalar(fem.form(self.jac*self.dx(i)))
            V = self.domain.comm.allreduce(V, op=MPI.SUM)
            self.V_subdomain.append(V)

    def calculate_areas_of_surfaces(self):
        """Fill ``A_area`` with integrated boundary areas per named boundary (MPI-reduced)."""
        self.A_area = []
        for i, bc_name in enumerate(self.bcs.keys(), 1):
            A = fem.assemble_scalar(fem.form(self.jac*self.ds(i)))
            A = self.domain.comm.allreduce(A, op=MPI.SUM)
            self.A_area.append(A)

    def get_subdomain_volume(self, domain):
        """Return cached volume for subdomain ``domain`` (name or integer tag index)."""
        i = self.compute_subdomain_index(domain)
        return self.V_subdomain[i]
    
    def get_surface_area(self, domain):
        """Return cached area for boundary ``domain`` (name or integer tag index)."""
        i = self.compute_boundary_index(domain)
        return self.A_area[i]
        
    def create_form_constants(self):
        """Define Crank–Nicolson weight ``theta``, time step ``dt``, and time ``t`` / ``t_n`` constants."""
        # constants
        self.theta = 0.5
        self.dt = fem.Constant(self.domain, PETSc.ScalarType((0.0)))
        self.t = fem.Constant(self.domain, PETSc.ScalarType((0.0)))
        self.t_n = fem.Constant(self.domain, PETSc.ScalarType((0.0)))

    def get_custom_data(self, key):
        """Return entry ``key`` from geometry metadata ``custom_data``."""
        return self.geo_meta['custom_data'][key]

    def compute_subdomain_index(self, domain):
        """Map subdomain integer tag or material name to a zero-based list index."""
        if np.issubdtype(type(domain), np.integer):
            return domain
        elif isinstance(domain, str):
            return self.subdomain_map[domain]
        
    def compute_boundary_index(self, boundary):
        """Map boundary integer tag or boundary name to a zero-based list index."""
        if np.issubdtype(type(boundary), np.integer):
            return boundary
        elif isinstance(boundary, str):
            return self.bcs_map[boundary]

    def get_measure_dx(self, domain):
        """Return the ``dx`` measure restricted to subdomain ``domain``."""
        i = self.compute_subdomain_index(domain)
        return self.dx(i+1)
        
    def get_measure_ds(self, boundary):
        """Return the exterior ``ds`` measure on boundary ``boundary``."""
        i = self.compute_boundary_index(boundary)
        return self.ds(i+1)
    
    def get_measure_dS(self, boundary):
        """Return the interior facet ``dS`` measure on boundary ``boundary``."""
        i = self.compute_boundary_index(boundary)
        return self.dS(i+1)

    def set_source_term(self, object, domain):
        """Register a volumetric source :class:`.Term` on subdomain ``domain``."""
        i = self.compute_subdomain_index(domain)
        self.q_source[i] = object
        
    def get_source_term(self, domain):
        """Return the source term registered on ``domain``, if any."""
        i = self.compute_subdomain_index(domain)
        return self.q_source[i]
        
    def set_bc_term(self, object, boundary):
        """Register a boundary / surface :class:`.Term` on ``boundary``."""
        i = self.compute_boundary_index(boundary)
        self.bcs_terms[i] = object
        
    def get_bc_term(self, boundary):
        """Return the boundary term on ``boundary``, if any."""
        i = self.compute_boundary_index(boundary)
        return self.bcs_terms[i]
        
    def create_form_terms_presized_lists(self):
        """Pre-allocate ``q_source`` / ``bcs_terms`` lists and unsteady callback buckets."""
        self.q_source = [None]*len(self.mats)
        self.bcs_terms = [None]*len(self.bcs)
        self.unsteady_term_update_callbacks = []
        self.unsteady_term_next_step_callbacks = []
        self.unsteady_term_converged_callbacks = []

    def update_unsteady_form_terms(self, t):
        """Call each registered Term ``update`` callback at simulation time ``t``."""
        for update_function in self.unsteady_term_update_callbacks:
            update_function(t)

    def next_step_unsteady_form_terms(self):
        """Call each Term ``next_step`` callback after advancing discrete time."""
        for next_step_function in self.unsteady_term_next_step_callbacks:
            next_step_function()

    def next_event(self):
        """Next special time event; default is no event (infinity). Overridable by subclasses."""
        return float("Inf")

    def converged_unsteady_form_terms(self):
        """Return True if every Term ``converged`` callback reports convergence (logical AND)."""
        all_converged = True
        for converged_function in self.unsteady_term_converged_callbacks:
            all_converged *= converged_function()
        return all_converged

    def predict_time_step_size(self):
        """Optional override for event-based ``dt`` prediction; default disables (infinity)."""
        return float('Inf')
    
    def get_unsteady_adaptive_time_step_size(self, dt_ctrl_interval):
        """Suggest next ``dt`` from max temperature change and Term ``adaptation`` factors."""
        # max T change in domain adaptation
        diff = self.T.x.petsc_vec - self.T_n.x.petsc_vec
        max_T_diff = np.abs(diff.array).max()
        max_T_diff = self.domain.comm.allreduce((max_T_diff), op=MPI.MAX)
        if max_T_diff > dt_ctrl_interval[1]:
            ref_T = 0.2*dt_ctrl_interval[0] + 0.8*dt_ctrl_interval[1]
            new_dt = self.dt.value * (ref_T/max_T_diff)
        elif max_T_diff < dt_ctrl_interval[0]:
            new_dt = self.dt.value / 0.95
        else:
            new_dt = self.dt.value

        # evaluate lowest dt from all form Terms
        for adaptation_function in self.unsteady_term_adaptation_callbacks:
            new_dt = min(new_dt, self.dt.value*adaptation_function())
        return new_dt

    def define_form_subdomain_terms(self):
        """Hook for subclasses to attach source and boundary terms to the weak form."""
        pass

    def resolve_term_callbacks(self, callback_name):
        """Collect unique callables ``getattr(term, callback_name)`` from all Terms."""
        unique_callbacks = []
        # collect all updaters for source terms
        for obj in self.q_source:
            if obj is not None and hasattr(obj, callback_name) and not getattr(obj, callback_name) in unique_callbacks:
                unique_callbacks.append(getattr(obj, callback_name))

        # collect all updaters for boundary terms
        for obj in self.bcs_terms:
            if obj is not None and hasattr(obj, callback_name) and not getattr(obj, callback_name) in unique_callbacks:
                unique_callbacks.append(getattr(obj, callback_name))

        return unique_callbacks

    def resolve_form_terms_update_callbacks(self):
        """Wire :meth:`update_unsteady_form_terms` to every Term's ``update`` method."""
        self.unsteady_term_update_callbacks = self.resolve_term_callbacks('update')

    def resolve_form_terms_next_step_callbacks(self):
        """Wire :meth:`next_step_unsteady_form_terms` to every Term's ``next_step``."""
        self.unsteady_term_next_step_callbacks = self.resolve_term_callbacks('next_step')

    def resolve_form_terms_converged_callbacks(self):
        """Wire :meth:`converged_unsteady_form_terms` to every Term's ``converged``."""
        self.unsteady_term_converged_callbacks = self.resolve_term_callbacks('converged')

    def resolve_form_terms_adaptation_callbacks(self):
        """Collect Term ``adaptation`` callbacks for time-step control."""
        self.unsteady_term_adaptation_callbacks = self.resolve_term_callbacks('adaptation')

    def get_all_forcing_form_terms(self, T, t):
        """Assemble diffusion, sources, and boundary contributions in the temperature residual."""
        Fd = 0
        for i, mat in enumerate(self.mats, 1):
            domain_name = mat.name
            # derivative term: 0 = dT/dt*rho(T)*cp(T)
            #Fd += mat.rho(T)*mat.cp(T)*self.dT*self.T_v*self.jac*self.dx(i)

            # unsteady state form heat capacity term: 0 = dT/dt*rho*cp 
            Fd += ufl.dot(mat.k(T)*ufl.grad(T), ufl.grad(self.T_v))*self.jac*self.dx(i)

            #internal sources of heat in domains
            if self.q_source[i-1] is not None: # if defined for a subdomain
                Fd += -self.q_source[i-1](T, self.x, t, domain=domain_name)*self.T_v*self.jac*self.dx(i)

        # external dynamical boundary conditions
        for i, bc in enumerate(self.bcs, 1): # i == subdomain index and measure index
            bc_name = bc
            if self.bcs_terms[i-1] is not None: # if defined for a surface
                Fd += self.bcs_terms[i-1](T, self.x, t, domain=bc_name)*self.T_v*self.jac*self.ds(i) 
        return Fd

    def create_time_derivative_form_solver(self):
        """(Experimental) Build a Newton solver for an intermediate dT/dt projection step."""
        self.Fd = 0.0
        # TODO: complete and test this so we can use RK4 time projection before
        #       Crank Nicolson step in unsteady sim
        for i, mat in enumerate(self.mats, 1):
            self.Fd += mat.rho(self.T_n)*mat.cp(self.T_n)*self.dT*self.T_v*self.jac*self.dx(i)
        self.Fd += self.get_all_forcing_form_terms(self.T_n, self.t_n)
        self.derivative_solver = self.create_newton_solver(self.Fd, self.dT, petsc_options_prefix="hbderivative_")

    def create_steady_state_form_solver(self):
        """Build the steady nonlinear problem ``Fss(T)=0`` and its SNES solver."""
        # steady state form: 0 = 0
        self.Fss = 0
        self.Fss += self.get_all_forcing_form_terms(self.T, t=None)
        self.steady_solver = self.create_newton_solver(self.Fss, self.T, petsc_options_prefix="hbsteady_")

    def create_unsteady_form_solver(self):
        """Build the Crank–Nicolson semidiscrete residual ``F`` and SNES solver for ``T``."""
        #TODO: Try using h(T) instead of rho(T)*cp(T)*T
        #unsteady state form: 0 = 0
        self.F = 0
        for i, mat in enumerate(self.mats, 1): # i == subdomain index / measure index
            domain_name = mat.name
            # unsteady state form heat capacity term: 0 = dT/dt*rho*cp
            self.F += mat.rho(self.T)*mat.cp(self.T)*self.T*self.T_v*self.jac*self.dx(i)
            self.F += -mat.rho(self.T_n)*mat.cp(self.T_n)*self.T_n*self.T_v*self.jac*self.dx(i)

        self.F += self.theta*self.dt*self.get_all_forcing_form_terms(self.T, self.t)
        self.F += (1-self.theta)*self.dt*self.get_all_forcing_form_terms(self.T_n, self.t_n)
        
        self.unsteady_solver = self.create_newton_solver(self.F, self.T, petsc_options_prefix="hbunsteady_")

    def create_forms_for_calculating_temperature_spectrum(self):
        """Preassemble forms for logit-smoothed temperature CDF / PDF per subdomain."""
        # greek-Psi forms for calculating cumulative temperature spectrum of subdomains
        self.T_hat = fem.Constant(self.domain, PETSc.ScalarType((1.0)))
        self.b = fem.Constant(self.domain, PETSc.ScalarType((1.0)))
        self.psi_forms = []
        self.psi_prime_forms = []
        for i, mat in enumerate(self.mats, 1):
            #cumulative forms
            self.psi_forms.append(fem.form(1/(1+ufl.exp(self.b*(self.T-self.T_hat)))/self.V_subdomain[i-1]*self.jac*self.dx(i)))

            # density forms
            self.psi_prime_forms.append(fem.form((self.b*ufl.exp(self.b*(self.T-self.T_hat)))/(1+ufl.exp(self.b*(self.T-self.T_hat)))**2/self.V_subdomain[i-1]*self.jac*self.dx(i)))

    def create_common_probes(self, probes):
        """Extension point for probes shared between steady and unsteady runs."""
        pass
        
    def create_steady_probes(self, probes):
        """Register built-in steady-state diagnostic probes on ``probes``."""

        @probes.register_probe('local_timestamp', 's')
        def local_timestamp():
            return time.time()

        @probes.register_probe('t_cpu', 's')
        def t_cpu():
            return self.elapsed_steady

        # Iteration counters are snapshotted from SNES inside
        # _solve_and_record() right after solve() returns; the probes just
        # surface the cached integers. This avoids querying PETSc at probe
        # time (which can return stale or zero values depending on solver
        # state).
        @probes.register_probe('NLS_iter', '-', format='d')
        def NLS_iter():
            return getattr(self, 'steady_nls_iter', 0)

        @probes.register_probe('KSP_iter', '-', format='d')
        def KSP_iter():
            return getattr(self, 'steady_ksp_iter', 0)

        @probes.register_probe('r_norm', '-')
        def Residual_norm():
            return self.steady_solver.solver.getFunctionNorm()

        self.create_common_probes(probes)

    def create_unsteady_probes(self, probes):
        """Register built-in transient diagnostic probes on ``probes``."""

        @probes.register_probe('local_timestamp', 's')
        def local_timestamp():
            return time.time()

        @probes.register_probe('progress', '%')
        def progress():
            return 100*self.t.value/self.t_max
        
        @probes.register_probe('dt', 's')
        def dt_value():
            return float(self.dt.value)
        
        @probes.register_probe('t_remain', 's')
        def remain_t():
            return (self.t_max - self.t.value)/self.dt.value*self.ielapsed
        
        @probes.register_probe('t_remain_avg', 's')
        def remain_t():
            t_cpu = probes.get_value('t_cpu')
            progress = probes.get_value('progress')
            return t_cpu/(progress)*(100-progress)
        
        @probes.register_probe('t_timestamp', 's')
        def t_timestamp():
            return self.timestamp_start + self.t.value
        
        @probes.register_probe('t_sim', 's')
        def t_sim():
            return float(self.t.value)
        
        @probes.register_probe('t_cpu', 's')
        def t_cpu():
            return self.elapsed
        
        @probes.register_probe('t_i', 's')
        def t_i():
            return self.ielapsed
        
        # Iteration counters: see note in create_steady_probes. The values
        # are captured by _solve_and_record() in the time-stepping loop.
        # `NLS_iter`/`KSP_iter` reflect only the *last* SNES solve in the
        # step (matches the previous semantics); `NLS_iter_step` /
        # `KSP_iter_step` accumulate across all attempted SNES solves of
        # the current step (including failed retries / dt-halving), giving
        # a more honest picture of the work spent.
        @probes.register_probe('NLS_iter', '-', format='d')
        def NLS_iter():
            return getattr(self, 'unsteady_nls_iter', 0)

        @probes.register_probe('KSP_iter', '-', format='d')
        def KSP_iter():
            return getattr(self, 'unsteady_ksp_iter', 0)

        @probes.register_probe('NLS_iter_step', '-', format='d')
        def NLS_iter_step():
            return getattr(self, 'unsteady_nls_iter_step', 0)

        @probes.register_probe('KSP_iter_step', '-', format='d')
        def KSP_iter_step():
            return getattr(self, 'unsteady_ksp_iter_step', 0)

        @probes.register_probe('TERM_iter', '-', format='d')
        def Term_iter():
            return self.unsteady_term_its
        
        @probes.register_probe('total_iter', '-', format='d')
        def Total_iter():
            return self.unsteady_total_iter
        
        @probes.register_probe('r_norm', '-')
        def Residual_norm():
            return self.unsteady_solver.solver.getFunctionNorm()
        
        self.create_common_probes(probes)

    def create_unsteady_probe_writer(self):
        """Instantiate ``self.unsteady_probes`` and populate it via :meth:`create_unsteady_probes`."""
        # probe writer
        self.unsteady_probes = Probe_writer()
        self.create_unsteady_probes(self.unsteady_probes)
    
    def create_steady_probe_writer(self):
        """Instantiate ``self.steady_probes`` and populate it via :meth:`create_steady_probes`."""
        # probe writer
        self.steady_probes = Probe_writer()
        self.create_steady_probes(self.steady_probes)

    def create_static_vtk_data(self):    
        """Gather mesh topology on rank 0 for PyVista plotting (``root_cells``, ``root_x``, ...)."""
        # vtk plot stuff that do not need to be recreated every time
        cells, types, x = plot.vtk_mesh(self.V)
        num_cells_local = self.domain.topology.index_map(self.domain.topology.dim).size_local
        num_dofs_local = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        self.num_dofs_local = num_dofs_local
        num_dofs_per_cell = cells[0]
        cells_dofs = (np.arange(len(cells)) % (num_dofs_per_cell+1)) != 0
        global_dofs = self.V.dofmap.index_map.local_to_global(cells[cells_dofs].copy())
        cells[cells_dofs] = global_dofs
        root = 0
        global_cells = self.domain.comm.gather(cells[:(num_dofs_per_cell+1)*num_cells_local], root=root)
        global_types = self.domain.comm.gather(types[:num_cells_local])
        global_x = self.domain.comm.gather(x[:self.V.dofmap.index_map.size_local,:], root=root)
        if MPI.COMM_WORLD.rank == 0:
            self.root_x = np.vstack(global_x)
            self.root_cells = np.concatenate(global_cells)
            self.root_types = np.concatenate(global_types)

    def create_newton_solver(self, F, u, petsc_options_prefix):
        """Build a ``dolfinx.fem.petsc.NonlinearProblem`` with default SNES/KSP options."""

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1e-6,
            "snes_rtol": 1e-6,
            #"snes_stol": 1e-6,
            "snes_max_it": 30,
            #"snes_monitor": None,
            "ksp_error_if_not_converged": "true",
            "snes_error_if_not_converged": "true",
            "ksp_rtol": 1e-10,
            #"ksp_atol": 1e-16,
            "ksp_max_it": 10000,
            "ksp_reuse_preconditioner": 'false',
            #"ksp_monitor": None,
            # "ksp_type": "gmres",
            # "pc_type": "gamg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_cycle_type": "v",
        }
        problem = dolfinx.fem.petsc.NonlinearProblem(
            F, 
            u,
            bcs=[],
            petsc_options=petsc_options,
            petsc_options_prefix=petsc_options_prefix,
            )


        return problem

    def _solve_and_record(self, problem, kind):
        """Run a SNES solve via dolfinx ``NonlinearProblem`` and snapshot the
        SNES/KSP iteration counts immediately, before anything else can
        mutate the underlying PETSc state.

        Stores per-solve and cumulative-per-step counters as attributes that
        probe callbacks can read without ever touching PETSc.

        Args:
            problem: dolfinx NonlinearProblem (must expose ``.solver``).
            kind: 'unsteady' or 'steady'. Selects which set of attributes
                to write to.

        Returns:
            Whatever ``problem.solve()`` returns.
        """
        result = problem.solve()
        snes = problem.solver
        # SNESGetIterationNumber: nonlinear iterations completed in the
        # most recent SNES solve.
        nls_iter = int(snes.getIterationNumber())
        # SNESGetLinearSolveIterations: total KSP iterations summed over
        # every Newton step of the most recent SNES solve.
        ksp_iter = int(snes.getLinearSolveIterations())
        # SNESConvergedReason: positive on convergence, negative on failure.
        # Stored so users can inspect why a solve failed without re-querying
        # SNES later (which can be misleading if state is reset).
        conv_reason = int(snes.getConvergedReason())

        if kind == 'unsteady':
            self.unsteady_nls_iter = nls_iter
            self.unsteady_ksp_iter = ksp_iter
            self.unsteady_snes_reason = conv_reason
            self.unsteady_nls_iter_step += nls_iter
            self.unsteady_ksp_iter_step += ksp_iter
        elif kind == 'steady':
            self.steady_nls_iter = nls_iter
            self.steady_ksp_iter = ksp_iter
            self.steady_snes_reason = conv_reason
        else:
            raise ValueError(f"unknown solver kind: {kind!r}")

        return result

    def _reset_step_iter_counters(self):
        """Zero the per-step cumulative iteration counters. Called once at
        the start of every accepted/attempted unsteady time step."""
        self.unsteady_nls_iter_step = 0
        self.unsteady_ksp_iter_step = 0
    
    def solve_time_derivative(self,
        dT_guess=None):
        """(Incomplete) Solve the auxiliary dT problem if ``create_time_derivative_form_solver`` is used."""
        if dT_guess is not None:
            self.T.x.array[:] = dT_guess
        self.r_derivative = self.derivative_sollver.solve(self.T)

    def solve_steady(
        self,
        T_guess=None,
        result_dir=None,
        probe_destinations=[],
        probes_callbacks=[],
        xdmf_file=None,
        save_probes=False,
        verbose=False,
        close_probes=True,
        atol=None,
        rtol=None,
        stol=None,
        ksp_rtol=None,
        h0_T_ref=None,
        ):

        """Solve the steady thermal balance once and optionally write probes / XDMF.

        Args:
            T_guess: Initial Newton guess for ``T`` (optional).
            result_dir: Directory for XDMF output when ``xdmf_file`` is set.
            probe_destinations: Probe writer destinations (CSV, DB, memory).
            probes_callbacks: Run after each probe evaluation batch.
            xdmf_file: If set, write ``T`` once to ``result_dir/xdmf_file``.
            save_probes: Reserved / unused in current implementation.
            verbose: If True, print probe table after the solve.
            close_probes: Whether to close the probe writer at the end (always True here).
            atol, rtol, stol, ksp_rtol: Nonlinear and linear solver tolerances.
            h0_T_ref: Enthalpy reference temperature for materials.

        Returns:
            The :class:`~heat_battery.simulations.probing.Probe_writer` used for this run.
        """
        # set initial guess for the solver
        if T_guess is not None:
            self.T.x.array[:] = T_guess

        # set materials zero enthalpy reference temperature
        if h0_T_ref is not None:
            self.mats.set_h0_T_ref(h0_T_ref)
        else:
            self.mats.set_h0_T_ref(20.0)

        # set tolerances
        if atol is not None:
            self.steady_solver.solver.atol = atol
        if rtol is not None:
            self.steady_solver.solver.rtol = rtol
        if stol is not None:
            self.steady_solver.solver.stol = stol
        if ksp_rtol is not None:
            self.steady_solver.solver.ksp.rtol = ksp_rtol
        # print(f"SNES Tolerances: {self.steady_solver.solver.getTolerances()}")
        # print(f"KSP Tolerances: {self.steady_solver.solver.ksp.getTolerances()}")

        # set unsteady probes outputs locations
        if probe_destinations:
            for destination in probe_destinations:
                self.steady_probes.add_destination(destination)
        self.steady_probes.initialize()

        if probes_callbacks is not None:
            self.steady_probes.set_callbacks(probes_callbacks)

        #FIXME: add update for source_terms and bc_terms before simulation
        #FIXME: this should run in loop when the Terms are implicit in nature
        #FIXME: this whole function will be unusable as of now, I focus on unsteady now!!! SORRY!
        T_backup = self.T.x.array.copy()
        try:
            self.r_steady = self._solve_and_record(self.steady_solver, kind='steady')
        except Exception as e:
            self.T.x.array[:] = T_backup
            self.steady_probes.close()
            self.steady_probes.reset_printer()
            raise e

        
        t_start = time.time()
        if MPI.COMM_WORLD.rank == 0:
            self.elapsed_steady = time.time() - t_start

        self.steady_probes.evaluate_probes()

        if verbose:
            self.steady_probes.pretty_print()

        self.steady_probes.write_all_set_probes_outputs()

        if xdmf_file is not None:
            xdmf = io.XDMFFile(self.domain.comm, os.path.join(result_dir, xdmf_file), "w")
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.T)
            xdmf.close()

        # properly close all outputs
        if xdmf_file is not None:
            xdmf.close()

        self.steady_probes.close()
        self.steady_probes.reset_printer()

        return self.steady_probes

    def solve_unsteady(self, 
            T0: float|None=None,
            T_guess: float|None=None,
            h0_T_ref: float|None=None,
            t_max: float=100,
            dt_start: float=1e-3,
            dt_min: float=1e-6,
            dt_max: float=1,
            dt_xdmf: float=0.1,
            dt_ctrl_interval: tuple[float, float]=(0.1, 0.25),
            atol: float|None=None,
            rtol: float|None=None,
            stol: float|None=None,
            ksp_rtol: float|None=None,
            verbose: bool=True,
            xdmf_file: str|None=None,
            result_dir: str|None=None,
            probe_destinations: list[dict[str, Any]]=[],
            probes_callbacks: list[Callable[[], None]]=[],
            force_explicit_terms: bool=False,
            max_term_its: int=3,
            use_time_projection: bool=True,
            call_backs: list[Callable[[], None]]=[],
            call_back_each_t: float|None=None,
            call_back_each_step: int|None=None,
            custom_xdmf_trigger: Callable[[], bool]|None=None,
            load_initial_checkpoint: str|None=None,
            checkpoint_dir: str|None=None,
            checkpoint_dt: float|None=None,
            checkpoint_callbacks: list[Callable[[], None]]=[],
            datetime_start: str|None=None,
            ):

        """
        Run the transient heat equation with adaptive time stepping and Term coupling.

        Advances ``t`` from zero (or from ``load_initial_checkpoint``) to ``t_max``,
        solves the nonlinear system each step, updates all Terms, and streams
        scalar probes plus optional XDMF temperature history.

        Args:
            T0: Uniform initial temperature when not loading a checkpoint.
            T_guess: Initial Newton guess when not loading a checkpoint.
            h0_T_ref: Reference temperature for material enthalpy.
            t_max: End time of the simulation (seconds).
            dt_start, dt_min, dt_max: Time-step bounds.
            dt_xdmf: Spacing in simulation time between full-field XDMF writes.
            dt_ctrl_interval: ``(low, high)`` band on max |ΔT| for step-size control.
            atol, rtol, stol, ksp_rtol: SNES/KSP tolerances.
            verbose: If True, print probe columns each step; else use a progress bar.
            xdmf_file: Base filename for XDMF output under ``result_dir`` (or None).
            result_dir: Output directory for XDMF (required if ``xdmf_file`` is set).
            probe_destinations: List of probe sink specs for :class:`Probe_writer`.
            probes_callbacks: Callables invoked after probes are evaluated.
            force_explicit_terms: Evaluate Term updates in explicit-only mode.
            max_term_its: Max outer iterations for implicit Term coupling per step.
            use_time_projection: Linear extrapolation guess for the next Newton solve.
            call_backs: Arbitrary callbacks triggered by ``call_back_each_t`` / step.
            call_back_each_t, call_back_each_step: Callback scheduling.
            custom_xdmf_trigger: Reserved (not implemented).
            load_initial_checkpoint: Directory with a prior ``save_unsteady_checkpoint``
                tree; restores state, probes, and optionally XDMF via snapshot.
            checkpoint_dir: Where to write periodic checkpoints when ``checkpoint_dt`` is set.
            checkpoint_dt: If set, save full state this often (simulation seconds).
            checkpoint_callbacks: Run after each checkpoint (e.g. upload to DB).
            datetime_start: ISO-like string anchoring probe wall-clock timestamps.

        Note:
            When both checkpoints and XDMF are enabled, domain files are snapshotted
            alongside each checkpoint so a resume matches CSV probe truncation.
        """
        
        if load_initial_checkpoint is None:
            # set initial state of simulation
            if T0 is not None:    
                self.T_n.x.array[:] = T0
            else:
                self.T_n.x.array[:] = 20.0

            # set initial guess for the solver
            if T_guess is not None:
                self.T.x.array[:] = T_guess
            elif T0 is not None:
                self.T.x.array[:] = T0
            else:
                self.T.x.array[:] = 20.0

            # set materials zero enthalpy reference temperature
            if h0_T_ref is not None:
                self.mats.set_h0_T_ref(h0_T_ref)
            else:
                self.mats.set_h0_T_ref(20.0)

            self.dt.value = dt_start
            in_event: bool = False
            pre_event_dt : float = 0.0
            self.elapsed = 0.0
            self.t.value = 0
            self.t_n.value = self.t.value
            prev_callback_t = 0.0
            prev_callback_step = 0
            next_xdmf_t = 0.0
            prev_checkpoint_t = 0.0
            step = 0
        elif True: #os.path.isfile(Path(load_initial_checkpoint).with_suffix('.bp')) and os.path.isfile(Path(load_initial_checkpoint).with_suffix('.json')):
            # set initial state of simulation from checkpoint
            # this function wills set the terms constants inplace.probe
            # some local data need to be handled by the caller manually.

            #TODO: do not mix in place and out of place 
            self.load_metadata_checkpoint(load_initial_checkpoint, checkpoint_fname='metadata')
            self.load_simulation_constants_checkpoint(load_initial_checkpoint, checkpoint_fname='constants') # load before functions
            self.load_function_checkpoint(load_initial_checkpoint, checkpoint_fname='functions')
            self.load_terms_checkpoint(load_initial_checkpoint, checkpoint_fname='terms')
            local_data = self.load_local_scope_data_checkpoint(load_initial_checkpoint, checkpoint_fname='local_data')
            in_event: bool = local_data["in_event"]
            pre_event_dt : float = local_data["pre_event_dt"]
            prev_callback_t = local_data["prev_callback_t"]
            prev_callback_step = local_data["prev_callback_step"]
            next_xdmf_t = local_data["next_xdmf_t"]
            prev_checkpoint_t = local_data["prev_checkpoint_t"]
            step = local_data["step"]
        # set tolerances
        if atol is not None:
            self.unsteady_solver.solver.atol = atol
        if rtol is not None:
            self.unsteady_solver.solver.rtol = rtol
        if stol is not None:
            self.unsteady_solver.solver.stol = stol
        if ksp_rtol is not None:
            self.unsteady_solver.solver.ksp.rtol = ksp_rtol
        # print(f"SNES Tolerances: {self.unsteady_solver.solver.getTolerances()}")
        # print(f"KSP Tolerances: {self.unsteady_solver.solver.ksp.getTolerances()}")

        if datetime_start is not None:
            self.timestamp_start = datetime.datetime.strptime(datetime_start, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
        else:
            self.timestamp_start = 0.0

        # set unsteady probes outputs locations
        if probe_destinations:
            for destination in probe_destinations:
                self.unsteady_probes.add_destination(destination)
        if load_initial_checkpoint is not None:
            self.load_probe_destination_checkpoints(load_initial_checkpoint, checkpoint_fname='probes')
        self.unsteady_probes.initialize()
        
        if probes_callbacks is not None:
            self.unsteady_probes.set_callbacks(probes_callbacks)

        self.unsteady_probes.evaluate_probes()
        self.update_unsteady_form_terms(self.t_n) #TODO: check automaticaly if this inner call calculates corectly and does not do divergence

        # open xdmf file for writing domain data
        # TODO: make this full domain outputs part of ProbeWriter for more dynamic definitions
        # TODO: check if adios4dolfinx can be used for this instead of io.XDMFFile
        if xdmf_file is not None:
            file_path = os.path.join(result_dir, xdmf_file)
            # When resuming from a checkpoint, restore the xdmf/.h5 snapshot
            # taken at that checkpoint and reopen in append mode so previously
            # written timesteps are preserved and new ones continue from
            # next_xdmf_t (also restored above). If no snapshot is available
            # we fall back to a fresh file -- this happens when xdmf output
            # was not enabled the previous run.
            if load_initial_checkpoint is not None:
                restored = self.restore_xdmf_files_from_checkpoint(
                    load_initial_checkpoint, file_path,
                )
                if restored:
                    xdmf = io.XDMFFile(self.domain.comm, file_path, "a")
                else:
                    self.print_r0(
                        f"WARNING: resuming from checkpoint at "
                        f"{load_initial_checkpoint} but no XDMF snapshot was "
                        "found there. Restarting XDMF output from scratch."
                    )
                    xdmf = io.XDMFFile(self.domain.comm, file_path, "w")
                    xdmf.write_mesh(self.domain)
            else:
                xdmf = io.XDMFFile(self.domain.comm, file_path, "w")
                xdmf.write_mesh(self.domain)

        if verbose:
            #self.unsteady_probes.pretty_string()
            printer = self.print_r0
            self.unsteady_probes.set_printer(printer)
        else:
            pbar = ProgressBar(
                desc = f"{self.__class__.__name__} unsteady simulation progress", 
                update_cb=lambda : self.unsteady_probes.get_value('progress'),
                )
            printer = pbar.print_message
            self.unsteady_probes.set_printer(printer)

        # time steping loop with time adaptation
        self.t_max = t_max
        stop_timesteping = False
        while self.t.value < self.t_max:
            i_start = time.time() # start stopwatch for cputime of one iteration
            step += 1
            success = False
            last_dt_min_atempt = False

            # Keep trying to solve for new time step
            self.unsteady_term_its = 1
            self.unsteady_total_iter = 0
            # Reset cumulative iteration counters for this new step. They
            # will be summed up across every SNES re-solve attempt in the
            # inner loop below.
            self._reset_step_iter_counters()
            while not success: # solving current time step
                self.unsteady_total_iter += 1
                
                # advance time in form
                if self.t_n.value + self.dt.value > self.t_max:
                    self.dt.value = self.t_max - self.t.value
                self.t.value = self.t_n.value + self.dt.value

                # check if event ahead crossed over, if so update time in form
                t_next_event = self.next_event()
                if self.t.value > t_next_event:
                    in_event = True
                    pre_event_dt = self.dt.value
                    self.dt.value = t_next_event - self.t_n.value
                    self.t.value = self.t_n.value + self.dt.value
                else:
                    in_event = False

                # Try to iteratively converge Terms withoud solving the PDE.
                # This is a bit of a hack to reduce the number of Nonlinear
                # solves. The idea is that the Terms might or might not be explicit,
                # if they are explicit we can evaluate them exactly with only 
                # one iteration (this will be detected automaticaly). If they are not
                # explicit (meaning they do not depend only on the PDE solution T
                # but also on oeach other) we can try to converge them with more
                # iterations. We can also force the explicit evaluation even if they
                # are not explicit (but this is likely to introduce consistency errors)
                # and potentialy save some cpu time. Recomended safe aprooach is to 
                # set `force_explicit_terms` to False.
                for i in range(3): #TODO, study impact of the int in the range
                    if not force_explicit_terms:
                        self.update_unsteady_form_terms(self.t)
                    if force_explicit_terms and self.converged_unsteady_form_terms():
                        break
                
                # try to solve the PDE (most cpu intensive code is in this chunk!)
                try:
                    self.r_unsteady = self._solve_and_record(self.unsteady_solver, kind='unsteady')
                except Exception as e:
                    if MPI.COMM_WORLD.rank == 0:
                        printer(
                            f"Nonlinear solver failed after "
                            f"{getattr(self, 'unsteady_nls_iter', 0)} NLS iterations and "
                            f"{getattr(self, 'unsteady_ksp_iter', 0)} KSP iterations "
                            f"(reason={getattr(self, 'unsteady_snes_reason', 'n/a')}) "
                            f"-> lowering time step by 30%: {e}"
                        )
                    self.dt.value *= 0.7 # lower the step after unexpected fail
                    self.T.x.array[:] = self.T_n.x.array[:]
                    self.unsteady_term_its = 1
                    if self.dt.value < dt_min:
                        stop_timesteping = True
                        break
                    continue
                    
                # Checks if Terms converged, if not we have to resolve the PDE again
                # because the data of PDE and the data of the Terms are inconsistent. 
                # If it takes too many atempts, lower time step and tell loop it is
                # not solving an event. If it was solving an event it will not be
                # after it lowers the dt since the event was detected at later time
                # point.
                self.unsteady_probes.evaluate_probes()
                if not force_explicit_terms and not self.converged_unsteady_form_terms():
                    self.unsteady_term_its += 1
                    if self.unsteady_term_its > max_term_its:
                        self.dt.value *= 0.5
                        self.dt.value = max(dt_min, self.dt.value)
                        if self.dt.value == dt_min and not last_dt_min_atempt:
                            last_dt_min_atempt = True
                        elif last_dt_min_atempt:
                            printer(f"Min dt is too large - Cannot converge Terms")
                            stop_timesteping = True
                            break
                        self.unsteady_term_its = 1
                        self.T.x.array[:] = self.T_n.x.array
                        printer(f"convergence stepdown -> dt: {self.dt.value}")
                        in_event = False
                    continue
                
                # Now PDE and Terms are consistent, but we have to check if there
                # is a variable that changed more than allowed by adaptation callbacks.
                # If this lowers the time step we have to solve the PDE again because
                # the solution is not valid anymore and inform the loop it is not solving
                # event at that point. If the new dt is larger than the current dt we
                # can continue since the solution is still valid and the increase in dt
                # will affect the solution at the next time step not the current one.
                new_dt = self.get_unsteady_adaptive_time_step_size(dt_ctrl_interval)
                if new_dt < self.dt.value:
                    self.dt.value = new_dt
                    if self.dt.value < dt_min and not last_dt_min_atempt:
                        self.dt.value = dt_min
                        last_dt_min_atempt = True
                    elif last_dt_min_atempt:
                        printer(f"Min dt is too large: dt <= {new_dt} required")
                        stop_timesteping = True
                        break
                    self.unsteady_term_its = 1
                    in_event = False
                    continue
                elif new_dt > self.dt.value and not self.unsteady_term_its >= max_term_its:
                    self.dt.value = new_dt
                    self.dt.value = min(dt_max, self.dt.value)

                # Now we check if it was solving event and if it was the dt could
                # have been very small, so lets put back in the pre-event dt to
                # keep the original step size and potentialy save some slow
                # backgrowth of the time step to its original size.
                if in_event:
                    self.dt.value = pre_event_dt
                    in_event = False
                
                success = True # stop the inner loop and write out all results
            
            # if inner loop stoped but this flag was set, stop the outer loop because
            # something went wrong and we need to stop the simulation completely.
            # This can happen if time step is smaller than the minimum allowed or the
            # solver failed too many times.
            if stop_timesteping:
                self.print_r0("Adaptive strategy triggered stop.")
                break
            
            # Save the domain data in specified time points defined by dt_xdmf.
            # This is a bit of a hack to save the domain data in specified time
            # points using linear extrapolation without the need to solve the PDE
            # at those points.
            denom_dt = self.t.value - self.t_n.value
            while next_xdmf_t <= self.t.value and xdmf_file is not None:
                enum_dt = next_xdmf_t - self.t_n.value
                self.T_temp.x.array[:] = self.T.x.array
                self.T_temp.x.array[:] -= self.T_n.x.array
                self.T_temp.x.array[:] *= enum_dt/denom_dt
                self.T_temp.x.array[:] += self.T_n.x.array
                xdmf.write_function(self.T_temp, next_xdmf_t)
                next_xdmf_t += dt_xdmf
            MPI.COMM_WORLD.Barrier()

            # Set initial guess for next Newton solve using linear extrapolation.
            # This might reduce NLS iterations and overall solve time.
            # TODO: try using some cubic stuff insted of linear extrapolation.
            #       it might save some NLS its.
            if use_time_projection:
                self.T_temp.x.array[:] = self.T.x.array
                self.T_temp.x.array[:] -= self.T_n.x.array
                self.T_temp.x.array[:] *= self.dt.value/denom_dt
                self.T_temp.x.array[:] += self.T_n.x.array
                self.T_n.x.array[:] = self.T.x.array
                self.t_n.value = self.t.value
                self.next_step_unsteady_form_terms()
                self.T.x.array[:] = self.T_temp.x.array
            else:
                self.T_n.x.array[:] = self.T.x.array
                self.t_n.value = self.t.value
                self.next_step_unsteady_form_terms()

            # Compute how long everything took
            if MPI.COMM_WORLD.rank == 0:
                self.ielapsed = time.time() - i_start
                self.elapsed += self.ielapsed

            # Write the probes (appending mode)
            self.unsteady_probes.write_all_set_probes_outputs()

            # Log step to terminal
            if verbose:
                self.unsteady_probes.pretty_print()
            elif MPI.COMM_WORLD.rank == 0:
                pbar.update()

            # Trigger callbacks
            if (call_back_each_t is not None) and (self.t.value > prev_callback_t + call_back_each_t):
                for call_back in call_backs:
                    call_back()
                prev_callback_t += call_back_each_t

            if (call_back_each_step is not None) and (step > prev_callback_step + call_back_each_step):
                for call_back in call_backs:
                    call_back()
                prev_callback_step += call_back_each_step

            # Save checkpoint if required
            if checkpoint_dt is not None and (self.t.value >= prev_checkpoint_t + checkpoint_dt):
                local_data = {
                    "prev_callback_t": prev_callback_t,
                    "prev_callback_step": prev_callback_step,
                    "next_xdmf_t": next_xdmf_t,
                    "prev_checkpoint_t": prev_checkpoint_t,
                    "in_event": in_event,
                    "pre_event_dt": pre_event_dt,
                    "step": step,
                }
                assert local_data["in_event"] == False, "Checkpoint should not be saved in event"
                prev_checkpoint_t += checkpoint_dt
                self.save_unsteady_checkpoint(checkpoint_dir, local_data=local_data)
                # Snapshot the XDMF output alongside the simulation state so
                # that a resume rolls the .xdmf/.h5 pair back to exactly this
                # point (mirrors how CSV probes are truncated to their
                # checkpointed byte size).
                if xdmf_file is not None:
                    xdmf.close()
                    self.save_xdmf_files_checkpoint(checkpoint_dir, file_path)
                    xdmf = io.XDMFFile(self.domain.comm, file_path, "a")
                for call_back in checkpoint_callbacks:
                    call_back()

        # properly close all outputs
        if xdmf_file is not None:
            xdmf.close()

        self.unsteady_probes.close()
        self.unsteady_probes.reset_printer()

        # # return the simulation results
        # if MPI.COMM_WORLD.rank == 0:
        #     return self.unsteady_probes

    def save_function_checkpoint(self, checkpoint_dir, checkpoint_fname='functions'):
        """Persist ``T_n`` and ``T`` with adios4dolfinx under ``checkpoint_dir``."""
        function_folder = Path(checkpoint_dir, checkpoint_fname)
        # save function of temperature T_n
        adios4dolfinx.write_function_on_input_mesh( 
            function_folder, 
            self.T_n, 
            mode=adios4dolfinx.adios2_helpers.adios2.Mode.Write,
            time=self.t_n.value, 
            name="Temperature_n",
        )

        # save function of temperature T_n
        adios4dolfinx.write_function_on_input_mesh( 
            function_folder, 
            self.T, 
            mode=adios4dolfinx.adios2_helpers.adios2.Mode.Append,
            time=self.t.value, 
            name="Temperature",
        )

    def save_simulation_constants_checkpoint(self, checkpoint_dir, checkpoint_fname='constants'):
        """Pickle scalar time state: ``elapsed``, ``t``, ``t_n``, ``dt``, ``h0_T_ref``."""
        data_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        data = {}
        data["elapsed"] = self.elapsed
        data["t_n"] = self.t_n.value
        data["t"] = self.t.value
        data["dt"] = self.dt.value
        data["h0_T_ref"] = self.mats[0].h0_T_ref.value
        save_data_binary(data_file, data)

    def save_terms_checkpoint(self, checkpoint_dir, checkpoint_fname='terms'):
        """Pickle :meth:`get_checkpoint_data` / :meth:`load_checkpoint_data` blobs for each Term."""
        terms_data_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        data = {}
        data["q_source_data"] = [None]*len(self.q_source)
        data["bcs_terms_data"] = [None]*len(self.bcs_terms)
        for i, term in enumerate(self.q_source):
            if term is not None:
                data[f"q_source_data"][i] = term.get_checkpoint_data()

        # get boundary Terms data
        for i, term in enumerate(self.bcs_terms):
            if term is not None:
                data[f"bcs_terms_data"][i] = term.get_checkpoint_data()

        save_data_binary(terms_data_file, data)

    def save_probe_destination_checkpoints(self, checkpoint_dir, checkpoint_fname='probes'):
        """Pickle per-destination state (e.g. CSV byte offsets) for ``unsteady_probes``."""
        probe_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        data = [None]*len(self.unsteady_probes.destinations)
        for i, probe_destination in enumerate(self.unsteady_probes.destinations):
            data[i] = probe_destination.get_checkpoint_data()

        save_data_binary(probe_file, data)

    def save_metadata_checkpoint(self, checkpoint_dir, checkpoint_fname='metadata'):
        """Write a small JSON sidecar with probe progress and wall time."""
        metadata_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".json")
        metadata = {
            "progress": self.unsteady_probes.get_value('progress'),
            "local_timestamp": datetime.datetime.now().isoformat(),
        }
        save_data_json(metadata_file, metadata)

    def save_local_scope_data_checkpoint(self, checkpoint_dir, checkpoint_fname='local_data', data=None):
        """Pickle the time-stepper's local loop variables (``next_xdmf_t``, ``step``, ...)."""
        data_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        save_data_binary(data_file, data)

    def save_unsteady_checkpoint(self, checkpoint_dir, local_data=None):
        """Full unsteady snapshot: constants, functions, terms, probes, local loop, metadata."""
        self.save_simulation_constants_checkpoint(checkpoint_dir, checkpoint_fname='constants') #must be first
        self.save_function_checkpoint(checkpoint_dir, checkpoint_fname='functions')
        self.save_terms_checkpoint(checkpoint_dir, checkpoint_fname='terms')
        self.save_probe_destination_checkpoints(checkpoint_dir, checkpoint_fname='probes')
        self.save_local_scope_data_checkpoint(checkpoint_dir, checkpoint_fname='local_data', data=local_data)
        self.save_metadata_checkpoint(checkpoint_dir, checkpoint_fname='metadata')

    @staticmethod
    def _xdmf_companion_paths(xdmf_file_path):
        """Return ``(xdmf_path, h5_path)`` for an XDMF output file."""
        xdmf_path = Path(xdmf_file_path)
        h5_path = xdmf_path.with_suffix(".h5")
        return xdmf_path, h5_path

    def save_xdmf_files_checkpoint(
            self,
            checkpoint_dir,
            xdmf_file_path,
            checkpoint_fname='xdmf',
        ):
        """Snapshot the live ``.xdmf`` / ``.h5`` pair into ``checkpoint_dir`` (rank 0 I/O).

        HDF5 cannot be safely byte-truncated like CSV; we copy the pair so a
        resume can restore consistent domain output. The XDMF file handle must
        be closed and flushed before calling this method.
        """
        if MPI.COMM_WORLD.rank == 0:
            xdmf_path, h5_path = self._xdmf_companion_paths(xdmf_file_path)
            backup_dir = Path(checkpoint_dir, checkpoint_fname)
            backup_dir.mkdir(parents=True, exist_ok=True)
            if xdmf_path.exists():
                shutil.copy2(xdmf_path, backup_dir / xdmf_path.name)
            if h5_path.exists():
                shutil.copy2(h5_path, backup_dir / h5_path.name)
        MPI.COMM_WORLD.Barrier()

    def restore_xdmf_files_from_checkpoint(
            self,
            checkpoint_dir,
            xdmf_file_path,
            checkpoint_fname='xdmf',
        ):
        """Inverse of :meth:`save_xdmf_files_checkpoint`.

        Copy the snapshot pair from ``checkpoint_dir`` on top of the live
        files, so that a subsequent open in append mode continues writing
        from the checkpointed state. Returns ``True`` if a backup was
        available and restored, ``False`` otherwise.
        """
        found = False
        if MPI.COMM_WORLD.rank == 0:
            xdmf_path, h5_path = self._xdmf_companion_paths(xdmf_file_path)
            backup_dir = Path(checkpoint_dir, checkpoint_fname)
            backup_xdmf = backup_dir / xdmf_path.name
            backup_h5 = backup_dir / h5_path.name
            if backup_xdmf.is_file() and backup_h5.is_file():
                xdmf_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_xdmf, xdmf_path)
                shutil.copy2(backup_h5, h5_path)
                found = True
        found = MPI.COMM_WORLD.bcast(found, root=0)
        MPI.COMM_WORLD.Barrier()
        return found

    def load_function_checkpoint(self, checkpoint_dir, checkpoint_fname='functions'):
        """Load ``T_n`` and ``T`` from an adios4dolfinx folder written by :meth:`save_function_checkpoint`."""
        function_folder = Path(checkpoint_dir, checkpoint_fname)
        adios4dolfinx.read_function(
            function_folder, 
            self.T_n, 
            time=self.t_n.value, 
            name="Temperature_n",
        )
        adios4dolfinx.read_function(
            function_folder, 
            self.T, 
            time=self.t.value, 
            name="Temperature",
        )

    def load_simulation_constants_checkpoint(self, checkpoint_dir, checkpoint_fname='constants'):
        """Restore scalar time / enthalpy state from :meth:`save_simulation_constants_checkpoint`."""
        data_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        data = load_data_binary(data_file)
        self.elapsed = data["elapsed"]
        self.t_n.value = data["t_n"]
        self.t.value = data["t"]
        self.dt.value = data["dt"]
        self.mats.set_h0_T_ref(data["h0_T_ref"])

    def load_terms_checkpoint(self, checkpoint_dir, checkpoint_fname='terms'):
        """Push pickled Term state into each non-None ``q_source`` / ``bcs_terms`` entry."""
        terms_data_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        data = load_data_binary(terms_data_file)
        # load source Terms data
        for i, term in enumerate(self.q_source):
            if term is not None:
                if data[f"q_source_data"][i] is not None:
                    term.load_checkpoint_data(data[f"q_source_data"][i])
                else:
                    mats_names = [mat.name for mat in self.mats]
                    raise ValueError(f"No data for source term[{i}]({mats_names[i]}) in checkpoint")

        # load boundary Terms data
        for i, term in enumerate(self.bcs_terms):
            if term is not None:
                if data[f"bcs_terms_data"][i] is not None:
                    term.load_checkpoint_data(data[f"bcs_terms_data"][i])
                else:
                    bcs_names = list(self.bcs.keys())
                    raise ValueError(f"No data for boundary term[{i}]({bcs_names[i]}) in checkpoint")

    def load_probe_destination_checkpoints(self, checkpoint_dir, checkpoint_fname='probes'):
        """Restore probe destination state (e.g. CSV truncation size) before :meth:`Probe_writer.initialize`."""
        data_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        data = load_data_binary(data_file)
        for i, probe_destination in enumerate(self.unsteady_probes.destinations):
            assert len(data) == len(self.unsteady_probes.destinations), "Number of probe destinations in checkpoint does not match"
            if data[i] is not None:
                probe_destination.load_checkpoint_data(data[i])
            else:
                probe_destination_names = list(probe_destination.keys())
                raise ValueError(f"No data for probe destination[{i}]({probe_destination_names[i]}) in checkpoint")

    def load_metadata_checkpoint(self, checkpoint_dir, checkpoint_fname='metadata'):
        """Read JSON metadata written by :meth:`save_metadata_checkpoint`."""
        metadata_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".json")
        metadata = load_data_json(metadata_file)
        return metadata
            
    def load_local_scope_data_checkpoint(self, checkpoint_dir, checkpoint_fname='local_data'):
        """Load the dict of time-stepper locals saved beside each checkpoint."""
        data_file = Path(checkpoint_dir, checkpoint_fname).with_suffix(".pickle")
        data = load_data_binary(data_file)
        return data

    def load_unsteady_checkpoint(self, checkpoint_dir):
        """Convenience wrapper: reload all unsteady checkpoint shards; returns ``local_data`` dict."""
        self.load_simulation_constants_checkpoint(checkpoint_dir, checkpoint_fname='constants')
        self.load_terms_checkpoint(checkpoint_dir, checkpoint_fname='terms')
        self.load_probe_destination_checkpoints(checkpoint_dir, checkpoint_fname='probes')
        self.load_function_checkpoint(checkpoint_dir, checkpoint_fname='functions')
        self.load_metadata_checkpoint(checkpoint_dir, checkpoint_fname='metadata')
        local_data = self.load_local_scope_data_checkpoint(checkpoint_dir, checkpoint_fname='local_data')
        return local_data

    def get_temperature_range(self, cell_tag=None):
        """Global min/max of ``T`` over ``cell_tag`` (or whole domain). Collective on all ranks."""
        if cell_tag is not None:
            cells = self.cell_tags.find(cell_tag)
            dofs = fem.locate_dofs_topological(self.V, 2, cells)
            T = self.T.x.array[dofs]
        else:
            T = self.T.x.array

        T_max = np.max(T, initial=-np.inf)
        T_max = self.domain.comm.allreduce((T_max), op=MPI.MAX)
        T_min = np.min(T, initial=np.inf)
        T_min = self.domain.comm.allreduce((T_min), op=MPI.MIN)
        return T_min, T_max
    
    def get_temperature_spectrum(self, T=None, cell_tag=None, sampling=1, smoothness=1, cumulative=False):
        """Evaluate the logit-smoothed temperature CDF (``cumulative``) or PDF on subdomain ``cell_tag``.

        Must be called collectively. If ``T`` is passed, temporarily swaps ``self.T`` for the scan.
        """
        #TODO: find more efficient method for calculationg this
        assert isinstance(cell_tag, int), "cell_tag must be integer"
        if T is not None:
            T_original = T.x.array.copy()
            self.T.x.array[:] = T.x.array
        T_min, T_max = self.get_temperature_range(cell_tag=cell_tag)
        if cumulative:
            psi = self.psi_forms[cell_tag-1]
        else:
            psi = self.psi_prime_forms[cell_tag-1]
        T_hat = np.arange(T_min, T_max, sampling)
        c = np.zeros_like(T_hat)
        self.b.value = smoothness 
        for i in range(len(T_hat)):
            self.T_hat.value = T_hat[i]
            c_value = fem.assemble_scalar(psi)
            c_value = self.domain.comm.allreduce(c_value, op=MPI.SUM) # ouch
            c[i] = c_value
        if T is not None:
            self.T.x.array[:] = T_original
        return T_hat, c

    def material_plot(self, m=None, property='k', include_density=False, T=None):
        """Plot material conductivity (or other ``property``) per subdomain; optionally overlay spectra."""
        m = m or range(len(self.mats))
        m = [m] if isinstance(m, int) else m
        figs = self.mats.plot_property(m=m, property=property)
        if include_density:
            for i, _m in enumerate(m):
                self.add_temperature_spectrum_trace(figs[i], m=_m, T=T)
        return figs

    def add_temperature_spectrum_trace(self, fig, m=None, T=None):
        """Append a secondary-axis spectrum trace to a Plotly ``fig`` (rank 0 only mutates ``fig``)."""
        T = T or [self.T]
        T = T if isinstance(T, list) else [T]
        for i, _T in enumerate(T):
            density_res = self.get_temperature_spectrum(T=_T, cell_tag=m+1)
            if MPI.COMM_WORLD.rank == 0:
                fig.add_trace(
                    go.Scatter(
                        x=density_res[0],
                        y=density_res[1], 
                        mode='lines', 
                        name=f"T spectrum ({i})", 
                        yaxis='y2'
                    )
                )
                fig.update_layout(     
                    yaxis2=dict(
                        title="Temperature spectrum [-]",
                        overlaying="y",
                        side="right"))
                
    def probes_time_plot(self, *args, **kwargs):
        """Line plot of ``unsteady_probes.df`` via Plotly Express (rank 0 only)."""
        if MPI.COMM_WORLD.rank == 0:
            fig = px.line(self.unsteady_probes.df, *args, **kwargs)
            fig.update_layout(
                uirevision="None",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    ),
                margin=dict(l=20, r=20, t=20, b=20),
                )
            return fig

    def domain_state_plot(self, T=None):
        """Build PyVista ``UnstructuredGrid``(s) with temperature on rank 0 (collective gather)."""
        root = 0
        T = T if T is not None else self.T
        T = T if isinstance(T, list) else [T]
        data = [None]*len(T)
        for i, _T in enumerate(T):
            global_vals = self.domain.comm.gather(_T.x.array[:self.num_dofs_local], root=root)

            if MPI.COMM_WORLD.rank == 0:
                root_vals = np.concatenate(global_vals)
                grid = pyvista.UnstructuredGrid(self.root_cells, self.root_types, self.root_x)
                grid.point_data["T"] = root_vals
                grid.set_active_scalars("T")
                data[i] = grid
        return data

    def get_initial_postprocess_data(self):
        """This method gets overloaded in the model. The default implementation 
        returns basic data about the simulation as a dictionary.
        """
        data = {}

        # add info about mesh and function space
        data['fem_space'] = {
            "dofs": self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs,
            }
        dim = self.domain.topology.dim
        for entity in ['cells', 'edges', 'faces', 'nodes']:
            self.domain.topology.create_entities(dim)
            data['fem_space'][entity] = self.domain.topology.index_map(dim).size_global
            dim -= 1
            if dim < 0:
                break
        
        # add info about subdomains
        data['subdomains'] = {} 
        for i, mat in enumerate(self.mats):
            volume = self.get_subdomain_volume(mat.name)
            mass = volume * mat.rho.evaluate(20)
            data['subdomains'][mat.name] = {
                "material": mat.__class__.__name__,
                "material_index": i,
                "volume": {"value": volume, "unit": "m^3"},
                "mass": {"value": mass, "unit": "kg"},
                "unit_price": {"value": mat.price, "unit": "EUR/kg"},
                "total_price": {"value": mass * mat.price, "unit": "EUR"},
                }
        data['total_subdomains_price'] = sum([d['total_price']['value'] for d in data['subdomains'].values()])
            
        # add info about surfaces
        data['surfaces'] = {}
        for i, bcs_name in enumerate(self.bcs):
            area = self.get_surface_area(bcs_name)
            data['surfaces'][bcs_name] = {
                "area": {"value": area, "unit": "m^2"},
                "price_per_square_meter": {"value": 0.0, "unit": "EUR/m^2"},
                "total_price": {"value": area * 0.0, "unit": "EUR"},
                }
        data['total_surfaces_price'] = {
            "value": sum([d['total_price']['value'] for d in data['surfaces'].values()]),
            "unit": "EUR",
            }
        data['total_price'] = {
            "value": data['total_subdomains_price'] + data['total_surfaces_price']['value'],
            "unit": "EUR",
            }
        return data

