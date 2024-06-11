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
from ..utilities import load_data

class Simulation():
    def __init__(self,
                geometry_dir='meshes/experiment', 
                result_dir='results/experiment_test', 
                model_name='mesh_2d',
                T0=20,
                T_guess=None, 
                t_max=3600,
                dt_start=0.01,
                dt_min=0.1,
                dt_max=60.0,
                dt_ctrl_interval=(0.1, 0.25),
                dt_xdmf=10,
                atol=1e-10,
                rtol=1e-12,
                h0_T_ref=20):
 
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
        self.create_xdmf_files()
        self.create_function_spaces()
        self.create_functions()
        self.calculate_volumes_of_subdomains()
        self.create_form_constants()
        self.create_form_terms_presized_lists()
        self.define_form_subdomain_terms()
        self.resolve_form_terms_update_callbacks()
        self.resolve_form_terms_next_step_callbacks()
        self.create_steady_state_form_solver()
        self.create_unsteady_form_solver()
        self.create_forms_for_calculating_temperature_spectrum()
        self.create_probe_writer()
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

    def create_xdmf_files(self):
        # create result files
        self.result_dir = os.path.join(self.result_dir, f'{self.dim}d')
        self.xdmf = io.XDMFFile(self.domain.comm, os.path.join(self.result_dir, 'functions.xdmf'), "w")
        self.xdmf.write_mesh(self.domain)
        self.xdmf_steady = io.XDMFFile(self.domain.comm, os.path.join(self.result_dir, 'steady_functions.xdmf'), "w")
        self.xdmf_steady.write_mesh(self.domain)

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

    def set_unsteady_source_term(self, object, domain):
        i = self.compute_subdomain_index(domain)
        self.q_source_unsteady[i] = object
        
    def get_unsteady_source_term(self, domain):
        i = self.compute_subdomain_index(domain)
        return self.q_source_unsteady[i]
        
    def set_unsteady_bc_term(self, object, boundary):
        i = self.compute_boundary_index(boundary)
        self.bcs_unsteady[i] = object
        
    def get_unsteady_bc_term(self, boundary):
        i = self.compute_boundary_index(boundary)
        return self.bcs_unsteady[i]
    
    def set_steady_state_source_term(self, object, domain):
        i = self.compute_subdomain_index(domain)
        self.q_source_steady[i] = object
        
    def get_steady_steady_source_term(self, domain):
        i = self.compute_subdomain_index(domain)
        return self.q_source_steady[i]
        
    def set_steady_state_bc_term(self, object, boundary):
        i = self.compute_boundary_index(boundary)
        self.bcs_steady[i] = object
        
    def get_steady_state_bc_term(self, boundary):
        i = self.compute_boundary_index(boundary)
        return self.bcs_steady[i]
        
    def create_form_terms_presized_lists(self):
        self.q_source_unsteady = [None]*len(self.mats)
        self.q_source_steady = [None]*len(self.mats)
        self.bcs_unsteady = [None]*len(self.bcs)
        self.bcs_steady = [None]*len(self.mats)
        self.unsteady_term_update_callbacks = []
        self.unsteady_term_next_step_callbacks = []

    def update_unsteady_form_terms(self, t):
        for update_function in self.unsteady_term_update_callbacks:
            update_function(t)

    def next_step_unsteady_form_terms(self):
        for next_step_function in self.unsteady_term_next_step_callbacks:
            next_step_function()

    def define_form_subdomain_terms(self):
        pass

    def resolve_form_terms_update_callbacks(self):
        # collect all updaters for unsteady volumetric terms
        for obj in self.q_source_unsteady:
            if obj is not None and not obj.update in self.unsteady_term_update_callbacks:
                self.unsteady_term_update_callbacks.append(obj.update)

        # collect all updaters for unsteady boundary terms
        for obj in self.bcs_unsteady:
            if obj is not None and not obj.update in self.unsteady_term_update_callbacks:
                self.unsteady_term_update_callbacks.append(obj.update)

    def resolve_form_terms_next_step_callbacks(self):
        # collect all next_steps for unsteady volumetric terms
        for obj in self.q_source_unsteady:
            if obj is not None and not obj.next_step in self.unsteady_term_next_step_callbacks:
                self.unsteady_term_next_step_callbacks.append(obj.next_step)

        # collect all updaters for unsteady boundary terms
        for obj in self.bcs_unsteady:
            if obj is not None and not obj.next_step in self.unsteady_term_next_step_callbacks:
                self.unsteady_term_next_step_callbacks.append(obj.next_step)

    def create_steady_state_form_solver(self):
        # steady state form: 0 = 0
        self.Fss = 0
        for i, mat in enumerate(self.mats, 1):
            domain_name = mat.name
            # steady state heat conduction term: 0 = λ*∇(∇(T))
            self.Fss += ufl.dot(mat.k(self.T)*ufl.grad(self.T), ufl.grad(self.T_v))*self.jac*self.dx(i)

            # stedy state source term: 0 = q(T)
            if self.q_source_steady[i-1] is not None:
                self.Fss += -self.q_source_steady[i-1](self.T, self.x, domain_name)*self.T_v*self.jac*self.dx(i)

        for i, bc in enumerate(self.bcs, 1):
            bc_name = bc
            if self.bcs_steady[i-1] is not None:
                self.Fss += self.bcs_steady[i-1](self.T, self.x, bc_name)*self.T_v*self.jac*self.ds(i)

        self.steady_solver = self.create_solver(self.Fss, self.T)

    def create_unsteady_form_solver(self):
        #unsteady state form: 0 = 0
        self.F = 0
        for i, mat in enumerate(self.mats, 1): # i == subdomain index / measure index
            domain_name = mat.name
            # unsteady state form heat capacity term: 0 = dT/dt*rho*cp
            self.F += mat.rho(self.T)*mat.cp(self.T)*self.T*self.T_v*self.jac*self.dx(i) 
            self.F += -mat.rho(self.T_n)*mat.cp(self.T_n)*self.T_n*self.T_v*self.jac*self.dx(i)

            # unsteady state heat conduction term: 0 = λ*∇(∇(T))
            self.F += self.theta*self.dt*ufl.dot(mat.k(self.T)*ufl.grad(self.T), ufl.grad(self.T_v))*self.jac*self.dx(i)
            self.F += (1-self.theta)*self.dt*ufl.dot(mat.k(self.T_n)*ufl.grad(self.T_n), ufl.grad(self.T_v))*self.jac*self.dx(i)

            # unstedy state source term: 0 = q(T)
            if self.q_source_unsteady[i-1] is not None: # if defined for a subdomain
                self.F += -self.theta*self.dt*self.q_source_unsteady[i-1](self.T, self.t, self.x, domain_name)*self.T_v*self.jac*self.dx(i) 
                self.F += -(1-self.theta)*self.dt*self.q_source_unsteady[i-1](self.T_n, self.t_n, self.x, domain_name)*self.T_v*self.jac*self.dx(i)

        for i, bc in enumerate(self.bcs, 1): # i == subdomain index / measure index
            bc_name = bc
            if self.bcs_unsteady[i-1] is not None: # if defined for a surface
                self.F += self.theta*self.dt*self.bcs_unsteady[i-1](self.T, self.t, self.x, bc_name)*self.T_v*self.jac*self.ds(i) 
                self.F += (1-self.theta)*self.dt*self.bcs_unsteady[i-1](self.T_n, self.t_n, self.x, bc_name)*self.T_v*self.jac*self.ds(i)
        
        self.unsteady_solver = self.create_solver(self.F, self.T)

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

    def create_probe_writer(self):
        # probe writer
        self.probes = Probe_writer(os.path.join(self.result_dir, 'probes.csv'))
        self.create_probes(self.probes)

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

    def create_solver(self, F, u):
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
    
    def close_results(self):
        self.probes.close()
        self.xdmf.close()
        self.xdmf_steady.close()
    
    def solve_steady(self, T_guess=None, save_xdmf=False):
        if T_guess is not None:
            self.T.x.array[:] = T_guess
        r = self.steady_solver.solve(self.T)
        self.probes.evaluate_probes()
        #self.probes.print()
        if save_xdmf:
            self.xdmf_steady.write_function(self.T)

    def solve_unsteady(self, 
            T0=None, T_guess=None, t_max=100, 
            verbose=False, save_xdmf=True, call_back=lambda: None, 
            call_back_each_t=None, call_back_each_step=None):
        self.t.value = 0
        prev_callback_t = 0.0
        prev_callback_step = 0
        self.t_n.value = self.t.value
        if T0 is not None:
            self.T_n.x.array[:] = T0
        if T_guess is not None:
            self.T.x.array[:] = T_guess
        t_start = time.time()
        next_xdmf_t = 0.0
        self.probes.evaluate_probes()
        self.update_unsteady_form_terms(self.t_n)
        if verbose:
            self.probes.print()
        self.t_max = t_max
        stop_timesteping = False
        step = 0
        while self.t.value < self.t_max:
            i_start = time.time()
            step += 1

            success = False
            last_dt_min_atempt = False
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
                    self.dt.value *= 0.5
                    self.T.x.array[:] = self.T_n.x.array[:]
                    continue

                # time step adaptation
                diff = self.T.vector - self.T_n.vector
                max_T_diff = np.abs(diff.array).max()
                max_T_diff = self.domain.comm.allreduce((max_T_diff), op=MPI.MAX)
                if max_T_diff > self.dt_ctrl_interval[1]:
                    self.dt.value *= 0.9
                    self.dt.value = max(self.dt_min, self.dt.value)
                    if self.dt.value == self.dt_min and not last_dt_min_atempt:
                        last_dt_min_atempt = True
                    elif last_dt_min_atempt:
                        print(f"Minimal dt is too large - max_T_diff was {max_T_diff}")
                        stop_timesteping = True
                        break
                    continue
                elif max_T_diff < self.dt_ctrl_interval[0]:
                    self.dt.value /= 0.95
                    self.dt.value = min(self.dt_max, self.dt.value)


                self.probes.evaluate_probes()
                success = True
            
            if stop_timesteping:
                print("Adaptive strategy triggered stop.")
                break

            while next_xdmf_t <= self.t.value and save_xdmf:
                denom_dt = self.t.value - self.t_n.value
                enum_dt = next_xdmf_t - self.t_n.value
                self.T_xdmf.x.array[:] = self.T.x.array
                self.T_xdmf.x.array[:] -= self.T_n.x.array
                self.T_xdmf.x.array[:] *= enum_dt/denom_dt
                self.T_xdmf.x.array[:] += self.T_n.x.array
                self.xdmf.write_function(self.T_xdmf, next_xdmf_t)
                next_xdmf_t += self.dt_xdmf
            MPI.COMM_WORLD.Barrier()

            self.T_n.x.array[:] = self.T.x.array
            self.t_n.value = self.t.value
            self.next_step_unsteady_form_terms()

            if MPI.COMM_WORLD.rank == 0:
                self.elapsed = time.time() - t_start
                self.ielapsed = time.time() - i_start
            self.probes.write_probes()

            if verbose:
                self.probes.print()

            if (call_back_each_t is not None) and (self.t.value > prev_callback_t + call_back_each_t):
                call_back()
                prev_callback_t += call_back_each_t

            if (call_back_each_step is not None) and (step > prev_callback_step + call_back_each_step):
                call_back()
                prev_callback_step += call_back_each_step

        if MPI.COMM_WORLD.rank == 0:
            return self.probes.df.copy()

    def create_probes(self, probes):

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
        
        @probes.register_probe('ksp_norm', '-')
        def KSP_norm():
            return self.unsteady_solver.krylov_solver.norm

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
            fig = px.line(self.probes.df, *args, **kwargs)
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
