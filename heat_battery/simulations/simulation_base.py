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

from .probing import Probe_writer
from ..materials import MaterialsSet
from ..utilities import save_data_binary, load_data_binary, ProgressBar, save_data_json, load_data_json

class Simulation():
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
        if MPI.COMM_WORLD.rank == 0:
            print(*args, **kwargs)

    def load_geometry(self):
        # """Fills attributes of the class with geometry data:
        #    dim: topological dimension of the mesh (1D, 2D, 3D)
        #    domain: mesh data
        #    cell_tags: integer tags of element-cell subdomain corespondence
        #    facet_tags: integer tags of element-boundary corespondance 
        #    mats: Material expresion set of named subdomains
        #    subdomain_map: dict for maping names to integer indices
        #    bcs: list of named boundaries
        #    bcs_map: 
        # """
        self.geo_path = os.path.join(self.geometry_dir, self.model_name)

        # load metadata
        self.geo_meta = load_data_binary(f'{self.geo_path}.ad')

        # load domain
        self.dim = self.geo_meta['dim']
        self.domain, self.cell_tags, self.facet_tags = io.gmshio.read_from_msh(f'{self.geo_path}.msh', MPI.COMM_WORLD, 0, gdim=self.dim)

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
        self.dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.dS = ufl.Measure("dS", domain=self.domain, subdomain_data=self.facet_tags)

    def create_function_spaces(self):
        #self.initial_condition = lambda x: np.full((x.shape[1],), self.T0)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))

    def create_functions(self):
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
        self.V_subdomain = []
        for i, mat in enumerate(self.mats, 1):
            V = fem.assemble_scalar(fem.form(self.jac*self.dx(i)))
            V = self.domain.comm.allreduce(V, op=MPI.SUM)
            self.V_subdomain.append(V)

    def calculate_areas_of_surfaces(self):
        self.A_area = []
        for i, bc_name in enumerate(self.bcs.keys(), 1):
            A = fem.assemble_scalar(fem.form(self.jac*self.ds(i)))
            A = self.domain.comm.allreduce(A, op=MPI.SUM)
            self.A_area.append(A)

    def get_subdomain_volume(self, domain):
        i = self.compute_subdomain_index(domain)
        return self.V_subdomain[i]
    
    def get_surface_area(self, domain):
        i = self.compute_boundary_index(domain)
        return self.A_area[i]
        
    def create_form_constants(self):
        # constants
        self.theta = 0.5
        self.dt = fem.Constant(self.domain, PETSc.ScalarType((0.0)))
        self.t = fem.Constant(self.domain, PETSc.ScalarType((0.0)))
        self.t_n = fem.Constant(self.domain, PETSc.ScalarType((0.0)))

    def get_custom_data(self, key):
        return self.geo_meta['call_data']['custom_data'][key]

    def compute_subdomain_index(self, domain):
        if np.issubdtype(type(domain), np.integer):
            return domain
        elif isinstance(domain, str):
            return self.subdomain_map[domain]
        
    def compute_boundary_index(self, boundary):
        if np.issubdtype(type(boundary), np.integer):
            return boundary
        elif isinstance(boundary, str):
            return self.bcs_map[boundary]

    def get_measure_dx(self, domain):
        i = self.compute_subdomain_index(domain)
        return self.dx(i+1)
        
    def get_measure_ds(self, boundary):
        i = self.compute_boundary_index(boundary)
        return self.ds(i+1)
    
    def get_measure_dS(self, boundary):
        i = self.compute_boundary_index(boundary)
        return self.dS(i+1)

    def set_source_term(self, object, domain):
        i = self.compute_subdomain_index(domain)
        self.q_source[i] = object
        
    def get_source_term(self, domain):
        i = self.compute_subdomain_index(domain)
        return self.q_source[i]
        
    def set_bc_term(self, object, boundary):
        i = self.compute_boundary_index(boundary)
        self.bcs_terms[i] = object
        
    def get_bc_term(self, boundary):
        i = self.compute_boundary_index(boundary)
        return self.bcs_terms[i]
        
    def create_form_terms_presized_lists(self):
        self.q_source = [None]*len(self.mats)
        self.bcs_terms = [None]*len(self.bcs)
        self.unsteady_term_update_callbacks = []
        self.unsteady_term_next_step_callbacks = []
        self.unsteady_term_converged_callbacks = []

    def update_unsteady_form_terms(self, t):
        for update_function in self.unsteady_term_update_callbacks:
            update_function(t)

    def next_step_unsteady_form_terms(self):
        for next_step_function in self.unsteady_term_next_step_callbacks:
            next_step_function()

    def next_event(self):
        return float("Inf")

    def converged_unsteady_form_terms(self):
        all_converged = True
        for converged_function in self.unsteady_term_converged_callbacks:
            all_converged *= converged_function()
        return all_converged

    def predict_time_step_size(self):
        return float('Inf')
    
    def get_unsteady_adaptive_time_step_size(self, dt_ctrl_interval):
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
        pass

    def resolve_term_callbacks(self, callback_name):
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
        "Collects update callbacks for all instances of Terms in unsteady form"
        self.unsteady_term_update_callbacks = self.resolve_term_callbacks('update')

    def resolve_form_terms_next_step_callbacks(self):
        "Collects next_steps callbacks for all instances of Terms in unsteady form"
        self.unsteady_term_next_step_callbacks = self.resolve_term_callbacks('next_step')

    def resolve_form_terms_converged_callbacks(self):
        "Collects converged callbacks for all instances of Terms in unsteady form"
        self.unsteady_term_converged_callbacks = self.resolve_term_callbacks('converged')

    def resolve_form_terms_adaptation_callbacks(self):
        "Collects converged callbacks for all instances of Terms in unsteady form"
        self.unsteady_term_adaptation_callbacks = self.resolve_term_callbacks('adaptation')

    def get_all_forcing_form_terms(self, T, t):
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
        self.Fd = 0.0
        # TODO: complete and test this so we can use RK4 time projection before
        #       Crank Nicolson step in unsteady sim
        for i, mat in enumerate(self.mats, 1):
            self.Fd += mat.rho(self.T_n)*mat.cp(self.T_n)*self.dT*self.T_v*self.jac*self.dx(i)
        self.Fd += self.get_all_forcing_form_terms(self.T_n, self.t_n)
        self.derivative_solver = self.create_newton_solver(self.dT, self.Fd)

    def create_steady_state_form_solver(self):
        # steady state form: 0 = 0
        self.Fss = 0
        self.Fss += self.get_all_forcing_form_terms(self.T, t=None)
        self.steady_solver = self.create_newton_solver(self.Fss, self.T)

    def create_unsteady_form_solver(self):
        #unsteady state form: 0 = 0
        self.F = 0
        for i, mat in enumerate(self.mats, 1): # i == subdomain index / measure index
            domain_name = mat.name
            # unsteady state form heat capacity term: 0 = dT/dt*rho*cp
            self.F += mat.rho(self.T)*mat.cp(self.T)*self.T*self.T_v*self.jac*self.dx(i)
            self.F += -mat.rho(self.T_n)*mat.cp(self.T_n)*self.T_n*self.T_v*self.jac*self.dx(i)

        self.F += self.theta*self.dt*self.get_all_forcing_form_terms(self.T, self.t)
        self.F += (1-self.theta)*self.dt*self.get_all_forcing_form_terms(self.T_n, self.t_n)
        
        self.unsteady_solver = self.create_newton_solver(self.F, self.T)

    def create_forms_for_calculating_temperature_spectrum(self):
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
        pass
        
    def create_steady_probes(self, probes):

        @probes.register_probe('local_timestamp', 's')
        def local_timestamp():
            return time.time()

        @probes.register_probe('t_cpu', 's')
        def t_cpu():
            return self.elapsed_steady

        @probes.register_probe('NLS_iter', '-', format='d')
        def NLS_iter():
            return self.r_steady[0]
        
        @probes.register_probe('KSP_iter', '-', format='d')
        def KSP_iter():
            return self.steady_solver.krylov_solver.its
        
        @probes.register_probe('ksp_norm', '-')
        def KSP_norm():
            return self.steady_solver.krylov_solver.norm

        self.create_common_probes(probes)

    def create_unsteady_probes(self, probes):

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
        
        @probes.register_probe('NLS_iter', '-', format='d')
        def NLS_iter():
            return self.r_unsteady[0]
        
        @probes.register_probe('KSP_iter', '-', format='d')
        def KSP_iter():
            return self.unsteady_solver.krylov_solver.its
        
        @probes.register_probe('TERM_iter', '-', format='d')
        def Term_iter():
            return self.unsteady_term_its
        
        @probes.register_probe('total_iter', '-', format='d')
        def Total_iter():
            return self.unsteady_total_iter
        
        @probes.register_probe('ksp_norm', '-')
        def KSP_norm():
            return self.unsteady_solver.krylov_solver.norm
        
        self.create_common_probes(probes)

    def create_unsteady_probe_writer(self):
        # probe writer
        self.unsteady_probes = Probe_writer()
        self.create_unsteady_probes(self.unsteady_probes)
    
    def create_steady_probe_writer(self):
        # probe writer
        self.steady_probes = Probe_writer()
        self.create_steady_probes(self.steady_probes)

    def create_static_vtk_data(self):    
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

    def create_newton_solver(self, F, u):
        problem = dolfinx.fem.petsc.NonlinearProblem(F, u)
        solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "residual"
        solver.max_it = 30
        solver.atol = 1e-6
        solver.rtol = 1e-6
        opts = PETSc.Options()
        ksp = solver.krylov_solver
        pc = ksp.getPC()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = 'gmres'
        opts[f"{option_prefix}pc_type"] = 'gamg'
        opts[f"{option_prefix}ksp_reuse_preconditioner"] = 'false'
        ksp.setFromOptions()
        return solver
    
    def solve_time_derivative(self,
        dT_guess=None):
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
        abs_tol=1e-6,
        rel_tol=1e-6,
        h0_T_ref=None,
        ):

        # set initial guess for the solver
        if T_guess is not None:
            self.T.x.array[:] = T_guess

        # set materials zero enthalpy reference temperature
        if h0_T_ref is not None:
            self.mats.set_h0_T_ref(h0_T_ref)
        else:
            self.mats.set_h0_T_ref(20.0)

        # set tolerances
        self.steady_solver.atol = abs_tol
        self.steady_solver.rtol = rel_tol

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
        self.r_steady = self.steady_solver.solve(self.T)
        
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
            T0=None,
            T_guess=None,
            h0_T_ref=None,
            t_max=100,
            dt_start=1e-3,
            dt_min=1e-6,
            dt_max=1,
            dt_xdmf=0.1,
            dt_ctrl_interval=(0.1, 0.25),
            atol=1e-5,
            rtol=1e-6,
            verbose=True,
            xdmf_file=None,
            result_dir=None,
            probe_destinations=[],
            probes_callbacks=[],
            force_explicit_terms=False,
            max_term_its=3,
            use_time_projection=True,
            call_backs=[],
            call_back_each_t=None,
            call_back_each_step=None,
            custom_xdmf_trigger=None,
            load_initial_checkpoint=None,
            checkpoint_dir=None,
            checkpoint_dt=None,
            checkpoint_callbacks=[],
            datetime_start=None,
            ):

        """
        This is the main method to run unstedy simulations. It handles the time 
        steping with adaptive time steping, advances and converges all Terms and
        writes out probes and domain data. It can write domain data (the solution) 
        to XDMF files and probes data to text files and/or a SQL database.

        Args:
            T0 (float): initial temperature in domain
            T_guess (np.ndarray): initial guess for the newton solver
            h0_T_ref (float): reference temperature for enthalpy
            t_max (float): maximum simulatated time
            dt_start (float): initial time step size
            dt_min (float): minimum time step size
            dt_max (float): maximum time step size
            dt_xdmf (float): time step for xdmf output
            dt_ctrl_interval (tuple of floats): temperature interval for adaptive time step control
            atol (float): absolute tolerance for the newton solver
            rtol (float): relative tolerance for the newton solver
            verbose (bool): print solver iterations
            xdmf_file (str): name of the xdmf file for domain data
            result_dir (str): directory for output files
            probes_file (str): name of the probes file for probe data
            result_database (str): name of the database for probe data
            database_table (str): name of the database table for probe data
            close_probes (bool): close probes file at the end of the simulation
            force_explicit_terms (bool): force explicit evaluation of the terms
            max_term_its (int): maximum number of term iterations to solve the terms
            use_time_projection (bool): use time projection for initial guess evaluations
            call_back (function): callback function called after specified iterations
            call_back_each_t (float): trigger callback after simulations advances this many seconds
            call_back_each_step (int): trigger callback after simulations advances this many steps
            custom_xdmf_trigger (function): not implemented yet
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
            self.t.value = 0
            self.t_n.value = self.t.value
            prev_callback_t = 0.0
            prev_callback_step = 0
            next_xdmf_t = 0.0
            prev_checkpoint_t = 0.0
        elif True: #os.path.isfile(Path(load_initial_checkpoint).with_suffix('.bp')) and os.path.isfile(Path(load_initial_checkpoint).with_suffix('.json')):
            # set initial state of simulation from checkpoint
            # this function wills set the terms constants inplace.
            # some local data need to be handled by the caller manually.
            local_data = self.load_unsteady_checkpoint(load_initial_checkpoint)
            in_event: bool = local_data["in_event"]
            pre_event_dt : float = local_data["pre_event_dt"]
            prev_callback_t = local_data["prev_callback_t"]
            prev_callback_step = local_data["prev_callback_step"]
            next_xdmf_t = local_data["next_xdmf_t"]
            prev_checkpoint_t = local_data["prev_checkpoint_t"]

        # set tolerances
        self.unsteady_solver.atol = atol
        self.unsteady_solver.rtol = rtol

        if datetime_start is not None:
            self.timestamp_start = datetime.datetime.strptime(datetime_start, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
        else:
            self.datetime_start = 0.0

        # start stopwatch for total cpu time
        t_start = time.time()

        # set unsteady probes outputs locations
        if probe_destinations:
            for destination in probe_destinations:
                self.unsteady_probes.add_destination(destination)
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
        step = 0
        while self.t.value < self.t_max:
            i_start = time.time() # start stopwatch for cputime of one iteration
            step += 1
            success = False
            last_dt_min_atempt = False

            # Keep trying to solve for new time step
            self.unsteady_term_its = 1
            self.unsteady_total_iter = 0
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
                    self.r_unsteady = self.unsteady_solver.solve(self.T)
                except Exception as e:
                    if MPI.COMM_WORLD.rank == 0:
                        printer(e)
                        #printer(f"Nonlinear solver failed after {self.r_unsteady[0]} NLS iterations and {self.unsteady_solver.krylov_solver.its} KSP iterations -> lowering time step by 30%")
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
                self.elapsed = time.time() - t_start
                self.ielapsed = time.time() - i_start

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
                }
                assert local_data["in_event"] == False, "Checkpoint should not be saved in event"
                prev_checkpoint_t += checkpoint_dt
                self.save_unsteady_checkpoint(checkpoint_dir, local_data=local_data)
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

    def save_unsteady_checkpoint(self, checkpoint_dir, local_data=None):

        function_folder = Path(checkpoint_dir, "functions")
        data_file = Path(checkpoint_dir, "data").with_suffix(".pickle")
        metadata_file = Path(checkpoint_dir, "metadata").with_suffix(".json")
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

        # get simulations constants
        data = {}
        data["t_n"] = self.t_n.value
        data["t"] = self.t.value
        data["dt"] = self.dt.value
        data["h0_T_ref"] = self.mats[0].h0_T_ref.value
        data["q_source_data"] = [None]*len(self.q_source) 
        data["bcs_terms_data"] = [None]*len(self.bcs_terms)
        data["local_data"] = {}
        if local_data is not None:
            for key, value in local_data.items():
                data["local_data"][key] = value

        # get source Terms data
        for i, term in enumerate(self.q_source):
            if term is not None:
                data[f"q_source_data"][i] = term.get_checkpoint_data()

        # get boundary Terms data
        for i, term in enumerate(self.bcs_terms):
            if term is not None:
                data[f"bcs_terms_data"][i] = term.get_checkpoint_data()

        # save data
        save_data_binary(data_file, data)

        # save metadata
        metadata = {
            "progress": self.unsteady_probes.get_value('progress'),
            "local_timestamp": datetime.datetime.now().isoformat(),
        }
        save_data_json(metadata_file, metadata)

    def load_unsteady_checkpoint(self, checkpoint_dir):
        function_folder = Path(checkpoint_dir, "functions")
        data_file = Path(checkpoint_dir, "data").with_suffix(".pickle")
        metadata_file = Path(checkpoint_dir, "metadata").with_suffix(".json")

        # load simulations constants
        data = load_data_binary(data_file)
        self.t_n.value = data["t_n"]
        self.t.value = data["t"]
        self.dt.value = data["dt"]
        self.mats.set_h0_T_ref(data["h0_T_ref"])

        # load metadata
        # metadata = load_data_json(metadata_file)

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
                
        # load function of temperature data
        adios4dolfinx.read_function(
            function_folder, 
            self.T_n, 
            time=self.t_n.value, 
            name="Temperature_n",
        )

        # load function of temperature data
        adios4dolfinx.read_function(
            function_folder, 
            self.T, 
            time=self.t.value, 
            name="Temperature",
        )

        return data["local_data"]

    def get_temperature_range(self, cell_tag=None):
        'This method must run on all rank to work properly'
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
        #TODO: find more efficient method for calculationg this
        'This method must run on all rank to work properly'
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
        'This method must run on all rank to work properly'
        m = m or range(len(self.mats))
        m = [m] if isinstance(m, int) else m
        figs = self.mats.plot_property(m=m, property=property)
        if include_density:
            for i, _m in enumerate(m):
                self.add_temperature_spectrum_trace(figs[i], m=_m, T=T)
        return figs

    def add_temperature_spectrum_trace(self, fig, m=None, T=None):
        '''This runs on all ranks, but mutates the fig only at rank=0'''
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
        'This method must run on all rank to work properly'
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

