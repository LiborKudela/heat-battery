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

from .probing import Probe_writer
from ..materials import MaterialsSet
from ..utilities import load_data, ProgressBar

class Simulation():
    def __init__(self,
                geometry_dir='meshes/experiment', 
                result_dir='results/experiment_test', 
                model_name='mesh_2d',
                T0=20,
                T_guess=None, 
                t_max=3600,
                dt_start=0.01,
                dt_min=0.01,
                dt_max=60.0,
                dt_ctrl_interval=(0.1, 0.25),
                dt_xdmf=10,
                atol=1e-10,
                rtol=1e-12,
                h0_T_ref=20,
                build_solvers=[
                    'derivative',
                    'steady',
                    'unsteady'],
                ):
 
        self.geometry_dir = geometry_dir
        self.result_dir = result_dir
        self.model_name = model_name
        self.T0 = T0
        self.T_guess = T_guess or self.T0
        self.t_max = t_max
        self.dt_start = dt_start
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_ctrl_interval = dt_ctrl_interval
        self.dt_xdmf = dt_xdmf
        self.atol = atol
        self.rtol = rtol
        self.h0_T_ref = h0_T_ref

        self.print_r0(f"Dolfinx version: {__version__}")
        self.load_geometry()
        self.define_measures()
        self.create_function_spaces()
        self.create_functions()
        self.calculate_volumes_of_subdomains()
        self.create_form_constants()
        self.create_form_terms_presized_lists()
        self.define_form_subdomain_terms()
        self.resolve_form_terms_update_callbacks()
        self.resolve_form_terms_next_step_callbacks()
        self.resolve_form_terms_converged_callbacks()

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
        self.geo_meta = load_data(f'{self.geo_path}.ad')

        # load domain
        self.dim = self.geo_meta['dim']
        self.domain, self.cell_tags, self.facet_tags = io.gmshio.read_from_msh(f'{self.geo_path}.msh', MPI.COMM_WORLD, 0, gdim=self.dim)

        # instantiate material*args, **kwargssSet(self.domain, self.geo_meta['materials'])
        self.mats = MaterialsSet(self.domain, self.geo_meta['materials'], self.h0_T_ref)
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
        self.T.x.array[:] = self.T_guess

        # temperature in previous time step
        self.T_n = fem.Function(self.V)
        self.T_n.name = "T_n"
        self.T_n.x.array[:] = self.T0

        # temperature time dirivative at state T_n and previous time t_n
        self.dT = fem.Function(self.V)
        self.dT.name = "dT"
        self.dT.x.array[:] = 0.0

        # temperature for interpolation (for equidistant time points)
        self.T_xdmf = fem.Function(self.V)
        self.T_xdmf.name = "T"
        self.T_xdmf.x.array[:] = self.T0

    def calculate_volumes_of_subdomains(self):
        self.V_subdomain = []
        for i, mat in enumerate(self.mats, 1):
            V = fem.assemble_scalar(fem.form(self.jac*self.dx(i)))
            V = self.domain.comm.allreduce(V, op=MPI.SUM)
            self.V_subdomain.append(V)

    def create_form_constants(self):
        # constants
        self.theta = 0.5
        self.dt = fem.Constant(self.domain, PETSc.ScalarType((self.dt_start)))
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

    def converged_unsteady_form_terms(self):
        all_converged = True
        for converged_function in self.unsteady_term_converged_callbacks:
            all_converged *= converged_function()
        return all_converged

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

        @probes.register_probe('progress', '%')
        def progress():
            return 100*self.t.value/self.t_max
        
        @probes.register_probe('dt', 's')
        def dt_value():
            return float(self.dt.value)
        
        @probes.register_probe('t_remain', 's')
        def remain_t():
            return (self.t_max - self.t.value)/self.dt.value*self.ielapsed
        
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
        def KSP_norm():
            return self.unsteady_term_its
        
        @probes.register_probe('ksp_norm', '-')
        def KSP_norm():
            return self.unsteady_solver.krylov_solver.norm
        
        self.create_common_probes(probes)

    def create_unsteady_probe_writer(self) -> Probe_writer:
        # probe writer
        result_dir = os.path.join(self.result_dir, f'{self.model_name}')
        self.unsteady_probes = Probe_writer(result_dir, 'unsteady.csv')
        self.create_unsteady_probes(self.unsteady_probes)
    
    def create_steady_probe_writer(self) -> Probe_writer:
        # probe writer
        result_dir = os.path.join(self.result_dir, f'{self.model_name}')
        self.steady_probes = Probe_writer(result_dir, 'steady_probes.csv')
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
        solver.atol = self.atol
        solver.rtol = self.rtol
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

    def solve_steady(self, 
        T_guess=None, xdmf_file=None, save_probes=False, 
        verbose=False, close_probes=True):
        t_start = time.time()
        if T_guess is not None:
            self.T.x.array[:] = T_guess

        #FIXME: add update for source_terms and bc_terms before simulation
        #FIXME: this should run in loop when the Terms are implicit in nature
        self.r_steady = self.steady_solver.solve(self.T)
        
        if MPI.COMM_WORLD.rank == 0:
            self.elapsed_steady = time.time() - t_start

        self.steady_probes.evaluate_probes()

        if verbose:
            self.steady_probes.pretty_print()

        if save_probes:
            self.steady_probes.write_probes_to_file()
        self.steady_probes.write_probes_to_memory()

        if xdmf_file is not None:
            xdmf = io.XDMFFile(self.domain.comm, os.path.join(self.result_dir, xdmf_file), "w")
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.T)
            xdmf.close()

        if close_probes:
            self.steady_probes.close()

        return self.steady_probes

    def solve_unsteady(self, 
            T0=None, T_guess=None, t_max=100, 
            verbose=True, xdmf_file=None, probes_file=None, close_probes=True,
            force_explicit_terms=False, max_term_its=3, use_time_projection=True, 
            call_back=lambda: None, call_back_each_t=None,
            call_back_each_step=None):

        # set initial state of simulation
        if T0 is not None:
            self.T_n.x.array[:] = T0
        else:
            self.T_n.x.array[:] = self.T0

        if T_guess is not None:
            self.T.x.array[:] = T_guess
        else:
            self.T.x.array[:] = self.T_guess
            
        t_start = time.time()
        self.t.value = 0
        prev_callback_t = 0.0
        prev_callback_step = 0
        self.t_n.value = self.t.value
        next_xdmf_t = 0.0

        # initialize unsteady probes and print to stdout
        if probes_file is not None:
            self.unsteady_probes.set_result_file_name(probes_file)
        self.unsteady_probes.evaluate_probes()
        self.update_unsteady_form_terms(self.t_n)

        # open xdmf file for writing domain data
        if xdmf_file is not None:
            xdmf = io.XDMFFile(self.domain.comm, os.path.join(self.result_dir, xdmf_file), "w")
            xdmf.write_mesh(self.domain)

        if verbose:
            self.unsteady_probes.pretty_string()
        elif MPI.COMM_WORLD.rank == 0:
            pbar = ProgressBar(
                desc = f"{self.__class__.__name__} unsteady simulation progress", 
                update_cb=lambda : self.unsteady_probes.get_value('progress'),
                )

        # time steping loop with time adaptation
        self.t_max = t_max
        stop_timesteping = False
        step = 0
        while self.t.value < self.t_max:
            i_start = time.time()
            step += 1
            success = False
            last_dt_min_atempt = False

            # Keep trying to solve for new time step
            self.unsteady_term_its = 1
            while not success:
                if self.t_n.value + self.dt.value > self.t_max:
                    self.dt.value = self.t_max - self.t.value
                self.t.value = self.t_n.value + self.dt.value

                self.update_unsteady_form_terms(self.t)

                try:
                    self.r_unsteady = self.unsteady_solver.solve(self.T)
                except Exception as e:
                    if MPI.COMM_WORLD.rank == 0:
                        print(e)
                        print(f"Nonlinear solver failed after {self.r_unsteady[0]} NLS iterations and {self.unsteady_solver.krylov_solver.its} KSP iterations -> lowering time step by 50%")
                    self.dt.value *= 0.5 # halve the step after unexpected fail
                    self.T.x.array[:] = self.T_n.x.array[:]
                    continue
                self.unsteady_probes.evaluate_probes()
                
                # converge any implicit Terms if needed
                if not force_explicit_terms and not self.converged_unsteady_form_terms():
                    self.unsteady_term_its += 1
                    continue
                high_term_its = self.unsteady_term_its >= max_term_its

                # Time step adaptation:
                diff = self.T.vector - self.T_n.vector
                max_T_diff = np.abs(diff.array).max()
                max_T_diff = self.domain.comm.allreduce((max_T_diff), op=MPI.MAX)
                if max_T_diff > self.dt_ctrl_interval[1]:
                    self.dt.value *= 0.9
                    self.dt.value = max(self.dt_min, self.dt.value)
                    if self.dt.value == self.dt_min and not last_dt_min_atempt:
                        last_dt_min_atempt = True
                    elif last_dt_min_atempt:
                        print(f"Min dt is too large - max_T_diff was {max_T_diff}")
                        stop_timesteping = True
                        break #stop loop that iterates on curent time step
                    continue
                elif max_T_diff < self.dt_ctrl_interval[0] and not high_term_its:
                    self.dt.value /= 0.95
                    self.dt.value = min(self.dt_max, self.dt.value)
                elif high_term_its:
                    self.dt.value *= 0.95
                    self.dt.value = max(self.dt_min, self.dt.value)
                
                success = True
            
            if stop_timesteping:
                print("Adaptive strategy triggered stop.")
                break # stop main unsteady solver loop
            
            denom_dt = self.t.value - self.t_n.value
            while next_xdmf_t <= self.t.value and xdmf_file is not None:
                enum_dt = next_xdmf_t - self.t_n.value
                self.T_xdmf.x.array[:] = self.T.x.array
                self.T_xdmf.x.array[:] -= self.T_n.x.array
                self.T_xdmf.x.array[:] *= enum_dt/denom_dt
                self.T_xdmf.x.array[:] += self.T_n.x.array
                xdmf.write_function(self.T_xdmf, next_xdmf_t)
                next_xdmf_t += self.dt_xdmf
            MPI.COMM_WORLD.Barrier()

            self.T_n.x.array[:] = self.T.x.array
            self.t_n.value = self.t.value
            self.next_step_unsteady_form_terms()

            if use_time_projection:
                self.T.x.array[:] -= self.T_n.x.array
                self.T.x.array[:] *= self.dt.value/denom_dt
                self.T.x.array[:] += self.T_n.x.array

            if MPI.COMM_WORLD.rank == 0:
                self.elapsed = time.time() - t_start
                self.ielapsed = time.time() - i_start

            self.unsteady_probes.write_probes_to_file()
            self.unsteady_probes.write_probes_to_memory()

            if verbose:
                self.unsteady_probes.pretty_print()
            elif MPI.COMM_WORLD.rank == 0:
                pbar.update()

            if (call_back_each_t is not None) and (self.t.value > prev_callback_t + call_back_each_t):
                call_back()
                prev_callback_t += call_back_each_t

            if (call_back_each_step is not None) and (step > prev_callback_step + call_back_each_step):
                call_back()
                prev_callback_step += call_back_each_step

        if xdmf_file is not None:
            xdmf.close()

        if close_probes:
            self.unsteady_probes.close()

        if MPI.COMM_WORLD.rank == 0:
            return self.unsteady_probes

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
                fig.add_trace(go.Scatter(x=density_res[0],
                                        y=density_res[1], 
                                        mode='lines', 
                                        name=f"T spectrum ({i})", 
                                        yaxis='y2'))
                fig.update_layout(     
                    yaxis2=dict(
                        title="Temperature spectrum [-]",
                        overlaying="y",
                        side="right"))
                
    def probes_time_plot(self, *args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            fig = px.line(self.unsteady_probes.df, *args, **kwargs)
            fig.update_layout(uirevision="None")
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
