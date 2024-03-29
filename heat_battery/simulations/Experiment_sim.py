from mpi4py import MPI
import sys
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

from .utilities import Probe_writer, FunctionSampler
from ..materials import MaterialsSet
from ..utilities import load_data

class Experiment():
    def __init__(self,
                geometry_dir='meshes/experiment', 
                result_dir='results/experiment_test', 
                dim=2, 
                T0=20,
                T_guess=None, 
                Qc=100,
                alpha = 7.0,
                t_max=3600,
                dt_start=0.1,
                dt_min=0.1,
                dt_max=60.0,
                dt_ctrl_interval=(0.1, 0.25),
                dt_xdmf=10,
                T_amb=20,
                regulation_step=0.01,
                T_cartridge_max = np.inf,
                atol=1e-10,
                rtol=1e-12):
 
        self.geometry_dir=geometry_dir
        self.result_dir = result_dir
        self.dim = dim
        self.T0 = T0
        self.T_guess = T_guess or self.T0
        self.Qc = Qc
        self.alpha = alpha
        self.t_max=t_max
        self.dt_start=dt_start
        self.dt_min=dt_min
        self.dt_max=dt_max
        self.dt_ctrl_interval=dt_ctrl_interval
        self.dt_xdmf=dt_xdmf
        self.T_amb=T_amb
        self.regulation_step=regulation_step
        self.T_cartridge_max = T_cartridge_max
        self.atol=atol
        self.rtol=rtol

        if MPI.COMM_WORLD.rank == 0:
            print(f"Dolfinx version: {__version__}")

        # load geometry
        self.geometry_path = os.path.join(geometry_dir, f'mesh_{dim}d')
        self.domain, self.cell_tags, self.facet_tags = io.gmshio.read_from_msh(f'{self.geometry_path}.msh', MPI.COMM_WORLD, 0, gdim=dim)
        geometry_ad_data = load_data(f'{self.geometry_path}.ad')

        # define measures for integration on subdomains
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)

        self.mats = MaterialsSet(self.domain, geometry_ad_data['materials'])
        bc_idx = geometry_ad_data['outer_surface_index']
        cartridge_index = geometry_ad_data['source_term_index']
        self.T_probes_coords = geometry_ad_data['probes_coords']
        self.T_probes_names = geometry_ad_data['probes_names'] 
        x = ufl.SpatialCoordinate(self.domain)
        jac = geometry_ad_data['jac_f'](x)

        # create result files
        self.result_dir = os.path.join(self.result_dir, f'{dim}d')
        self.xdmf = io.XDMFFile(self.domain.comm, os.path.join(self.result_dir, 'functions.xdmf'), "w")
        self.xdmf.write_mesh(self.domain)
        self.xdmf_steady = io.XDMFFile(self.domain.comm, os.path.join(self.result_dir, 'steady_functions.xdmf'), "w")
        self.xdmf_steady.write_mesh(self.domain)

        #self.initial_condition = lambda x: np.full((x.shape[1],), self.T0)
        self.V = fem.FunctionSpace(self.domain, ("Lagrange", 1))
        T_v = ufl.TestFunction(self.V)

        # temperature in current time step
        self.T = fem.Function(self.V)
        self.T.name = "T"
        #self.T.interpolate(self.initial_condition)
        self.T.x.array[:] = self.T_guess

        # temperature in previous time step
        self.T_n = fem.Function(self.V)
        self.T_n.name = "T_n"
        #self.T_n.interpolate(self.initial_condition)
        self.T_n.x.array[:] = self.T0

        # temperature for interpolation (for equidistant time points)
        self.T_xdmf = fem.Function(self.V)
        self.T_xdmf.name = "T"
        #self.T_xdmf.interpolate(self.initial_condition)
        self.T_xdmf.x.array[:] = self.T0

        self.V_subdomain = []
        for i, mat in enumerate(self.mats, 1):
            V_subdomain = fem.assemble_scalar(fem.form(jac*dx(i)))
            V_subdomain = self.domain.comm.allreduce(V_subdomain, op=MPI.SUM)
            self.V_subdomain.append(V_subdomain)

        self.Vc = self.V_subdomain[cartridge_index-1]
        self.q = fem.Constant(self.domain, PETSc.ScalarType((self.Qc/self.Vc)))
        self.q_n = fem.Constant(self.domain, PETSc.ScalarType((self.Qc/self.Vc)))

        # constants
        theta = 0.5
        self.T_amb = fem.Constant(self.domain, PETSc.ScalarType((T_amb)))
        self.T_amb_n = fem.Constant(self.domain, PETSc.ScalarType((T_amb)))
        self.dt = fem.Constant(self.domain, PETSc.ScalarType((dt_start)))

        # steady state form
        self.Fss = 0
        for i, mat in enumerate(self.mats, 1):
            self.Fss += ufl.dot(mat.k(self.T)*ufl.grad(self.T), ufl.grad(T_v))*jac*dx(i)
        self.Fss += alpha*(self.T - self.T_amb)*T_v*jac*ds(bc_idx)
        self.Fss += -self.q*T_v*jac*dx(cartridge_index)
        self.steady_solver = self.create_solver(self.Fss, self.T)

        #unsteady state form
        self.F = 0
        for i, mat in enumerate(self.mats, 1):
            self.F += mat.rho(self.T)*mat.cp(self.T)*self.T*T_v*jac*dx(i) - mat.rho(self.T_n)*mat.cp(self.T_n)*self.T_n*T_v*jac*dx(i)
            self.F += theta*self.dt*ufl.dot(mat.k(self.T)*ufl.grad(self.T), ufl.grad(T_v))*jac*dx(i) + (1-theta)*self.dt*ufl.dot(mat.k(self.T_n)*ufl.grad(self.T_n), ufl.grad(T_v))*jac*dx(i)
        # outer surface heat loss
        self.F += theta*self.dt*alpha*(self.T - self.T_amb)*T_v*jac*ds(bc_idx) + (1-theta)*self.dt*alpha*(self.T_n - self.T_amb_n)*T_v*jac*ds(bc_idx)
        # heated cartridge power term
        self.F += -theta*self.dt*self.q*T_v*jac*dx(cartridge_index) - (1-theta)*self.dt*self.q_n*T_v*jac*dx(cartridge_index)
        self.unsteady_solver = self.create_solver(self.F, self.T)

        # form for calculating heat in the whole domain
        h_form = 0
        for i, mat in enumerate(self.mats, 1):
            h_form += mat.h(self.T)*mat.rho(self.T)*jac*dx(i)
        self.H_form = fem.form(h_form)
        self.Qloss_form = fem.form(alpha*(self.T - self.T_amb)*jac*ds(bc_idx))

        # greek-Psi forms for calculating cumulative temperature density of subdomains
        self.T_hat = fem.Constant(self.domain, PETSc.ScalarType((1.0)))
        self.b = fem.Constant(self.domain, PETSc.ScalarType((1.0)))
        self.psi_forms = []
        self.psi_prime_forms = []
        for i, mat in enumerate(self.mats, 1):
            #cumulative forms
            self.psi_forms.append(fem.form(1/(1+ufl.exp(self.b*(self.T-self.T_hat)))/self.V_subdomain[i-1]*jac*dx(i)))

            # density forms
            self.psi_prime_forms.append(fem.form((self.b*ufl.exp(self.b*(self.T-self.T_hat)))/(1+ufl.exp(self.b*(self.T-self.T_hat)))**2/self.V_subdomain[i-1]*jac*dx(i)))

        # probe writer
        self.probes = Probe_writer(os.path.join(self.result_dir, 'probes.csv'))
        self.create_probes(self.probes)

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
    
    def solve_steady(self, Qc=100, T_amb=20, save_xdmf=True):
        self.T_amb.value = T_amb
        self.q.value = Qc/self.Vc
        r = self.steady_solver.solve(self.T)
        self.probes.evaluate_probes()
        #self.probes.print()
        if save_xdmf:
            self.xdmf_steady.write_function(self.T)

        return pd.Series(data=self.probes.get_value('T'), index=self.T_probes_names, name="Simulation")

    def solve_unsteady(self, Qc_t=None, T_amb_t=None, T0=None, t_max=100, verbose=False, save_xdmf=True, call_back=lambda: None, call_back_each=100):
        self.t = 0
        prev_callback_t = 0.0
        self.t_n = self.t
        self.T_amb.value = T_amb_t(self.t)
        self.q.value = Qc_t(self.t)/self.Vc
        if T0 is not None:
            self.T.x.array[:] = T0
            self.T_n.x.array[:] = T0
        t_start = time.time()
        next_xdmf_t = 0.0
        self.probes.evaluate_probes()
        if verbose:
            self.probes.print()
        self.t_max = t_max
        while self.t < self.t_max:
            i_start = time.time()

            success = False
            while not success:
                if self.t_n + self.dt.value > self.t_max:
                    self.dt.value = self.t_max - self.t
                self.t = self.t_n + self.dt.value

                self.T_amb.value = T_amb_t(self.t)
                self.q.value = Qc_t(self.t)/self.Vc
                self.T_amb_n.value = T_amb_t(self.t_n)
                self.q_n.value = Qc_t(self.t_n)/self.Vc
              
                self.r_unsteady = self.unsteady_solver.solve(self.T)
                # time step adaptation
                diff = self.T.vector - self.T_n.vector
                max_T_diff = np.abs(diff.array).max()
                max_T_diff = self.domain.comm.allreduce((max_T_diff), op=MPI.MAX)
                if max_T_diff > self.dt_ctrl_interval[1]:
                    self.dt.value *= 0.9
                    self.dt.value = max(self.dt_min, self.dt.value)
                    continue
                elif max_T_diff < self.dt_ctrl_interval[0]:
                    self.dt.value /= 0.95
                    self.dt.value = min(self.dt_max, self.dt.value)

                self.probes.evaluate_probes()
                success = True  

            while next_xdmf_t <= self.t and save_xdmf:
                denom_dt = self.t - self.t_n
                enum_dt = next_xdmf_t - self.t_n
                self.T_xdmf.x.array[:] = self.T.x.array
                self.T_xdmf.x.array[:] -= self.T_n.x.array
                self.T_xdmf.x.array[:] *= enum_dt/denom_dt
                self.T_xdmf.x.array[:] += self.T_n.x.array
                self.xdmf.write_function(self.T_xdmf, next_xdmf_t)
                next_xdmf_t += self.dt_xdmf
            MPI.COMM_WORLD.Barrier()

            if self.t > prev_callback_t + call_back_each:
                call_back()
                prev_callback_t += call_back_each

            self.T_n.x.array[:] = self.T.x.array
            self.t_n = self.t

            if MPI.COMM_WORLD.rank == 0:
                self.elapsed = time.time() - t_start
                self.ielapsed = time.time() - i_start
            self.probes.write_probes()
            if verbose:
                self.probes.print()

    def create_probes(self, probes):

        @probes.register_probe('progress', '%')
        def progress():
            return 100*self.t/self.t_max
        
        @probes.register_probe('dt', 's')
        def dt_value():
            return float(self.dt.value)
        
        @probes.register_probe('t_remain', 's')
        def remain_t():
            return (self.t_max - self.t)/self.dt.value*self.ielapsed
        
        @probes.register_probe('t_sim', 's')
        def t_sim():
            return self.t
        
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
        
        @probes.register_probe('heat', 'J')
        def H_probe():
            H = fem.assemble_scalar(self.H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H
        
        @probes.register_probe('heat loss', 'W')
        def loss_probe():
            q_flow = fem.assemble_scalar(self.Qloss_form)
            q_flow = self.domain.comm.allreduce((q_flow), op=MPI.SUM)
            return q_flow
        
        sampler = FunctionSampler(self.T_probes_coords, self.domain)
        @probes.register_probe('T', '°C')
        def Tc_probe():
            return sampler.eval(self.T)
        
        @probes.register_probe('power', 'W')
        def power():
            return self.q.value*self.Vc

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
        figs = []
        for i, _m in enumerate(m):
            # only rank=0 return fig, other return None
            fig = self.mats.plot_property(m=_m, property=property)
            figs.append(fig)

            if MPI.COMM_WORLD.rank == 0:
                figs[-1].update_layout(title=dict(text=self.mats[_m].name, x=0.5))

            if include_density:
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

    def domain_state_plot(self, T=None):
        'This method must run on all rank to work properly'
        root = 0
        T = T or self.T
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
