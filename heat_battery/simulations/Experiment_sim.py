from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, io, fem, nls, __version__, plot
import numpy as np
import ufl
import time
import os
import cloudpickle
import pyvista
import pandas as pd

from .utilities import Probe_writer, probe_function
from .. import materials
from ..data import Experiment_data

class Experiment():
    def __init__(self, geometry_dir='meshes/experiment', 
                result_dir='results/experiment_test', 
                dim=3, 
                T0=20, 
                Qc=100,
                alpha = 7.0,
                t_max=3600,
                dt_start=0.1,
                dt_min=0.1,
                dt_max=60.0,
                dt_xdmf=10,
                T_amb=20,
                regulation_step=0.01,
                T_cartridge_max = np.inf,
                atol=1e-10,
                rtol=1e-10):
        
        self.geometry_dir=geometry_dir
        self.result_dir = result_dir
        self.dim = dim
        self.T0 = T0
        self.Qc = Qc
        self.alpha = alpha
        self.t_max=t_max
        self.dt_start=dt_start
        self.dt_min=dt_min
        self.dt_max=dt_max
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

        # define measures for integration on subdomains
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)

        # load aditional data of the geometry
        with open(f'{self.geometry_path}.ad', 'rb') as fp:
            geometry_ad_data = cloudpickle.load(fp)
        mats = [eval(f'materials.{mat}') for mat in geometry_ad_data['materials']]
        mats_names = geometry_ad_data['materials_names']
        self.mats = [mat(self.domain, name=name) for mat, name in zip(mats, mats_names)]
        bc_idx = geometry_ad_data['outer_surface_index']
        cartridge_index = geometry_ad_data['cartridge_heated_index']
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

        self.initial_condition = lambda x: np.full((x.shape[1],), self.T0)
        self.V = fem.FunctionSpace(self.domain, ("CG", 1))
        T_v = ufl.TestFunction(self.V)

        # temperature in current time step
        self.T = fem.Function(self.V)
        self.T.name = "T"
        self.T.interpolate(self.initial_condition)

        # temperature in previous time step
        self.T_n = fem.Function(self.V)
        self.T_n.name = "T_n"
        self.T_n.interpolate(self.initial_condition)

        # temperature for interpolation (for equidistant time points)
        self.T_xdmf = fem.Function(self.V)
        self.T_xdmf.name = "T"
        self.T_xdmf.interpolate(self.initial_condition)

        self.Vc = fem.assemble_scalar(fem.form(jac*dx(cartridge_index)))
        self.Vc = self.domain.comm.allreduce((self.Vc), op=MPI.SUM)
        self.q = fem.Constant(self.domain, PETSc.ScalarType((self.Qc/self.Vc)))
        self.q_n = fem.Constant(self.domain, PETSc.ScalarType((self.Qc/self.Vc)))

        # constants
        theta = 0.5
        self.T_amb = fem.Constant(self.domain, PETSc.ScalarType((T_amb)))
        self.T_amb_n = fem.Constant(self.domain, PETSc.ScalarType((T_amb)))
        self.dt = fem.Constant(self.domain, PETSc.ScalarType((dt_start)))

        # steady state form
        Fss = 0
        for i, mat in enumerate(self.mats, 1):
            Fss += ufl.dot(mat.k(self.T)*ufl.grad(self.T), ufl.grad(T_v))*jac*dx(i)
        Fss += alpha*(self.T - self.T_amb)*T_v*jac*ds(bc_idx)
        Fss += -self.q*T_v*jac*dx(cartridge_index)
        self.steady_solver = self.create_solver(Fss, self.T)

        #unsteady state form
        F = 0
        for i, mat in enumerate(self.mats, 1):
            F += mat.rho(self.T)*mat.cp(self.T)*self.T*T_v*jac*dx(i) - mat.rho(self.T_n)*mat.cp(self.T_n)*self.T_n*T_v*jac*dx(i)
            F += theta*self.dt*ufl.dot(mat.k(self.T)*ufl.grad(self.T), ufl.grad(T_v))*jac*dx(i) + (1-theta)*self.dt*ufl.dot(mat.k(self.T_n)*ufl.grad(self.T_n), ufl.grad(T_v))*jac*dx(i)
        # outer surface heat loss
        F += theta*self.dt*alpha*(self.T - self.T_amb)*T_v*jac*ds(bc_idx) + (1-theta)*self.dt*alpha*(self.T_n - self.T_amb_n)*T_v*jac*ds(bc_idx)
        # heated cartridge power term
        F += -theta*self.dt*self.q*T_v*jac*dx(cartridge_index) - (1-theta)*self.dt*self.q_n*T_v*jac*dx(cartridge_index)
        self.unsteady_solver = self.create_solver(F, self.T)

        f = 0
        for i, mat in enumerate(self.mats, 1):
            f += mat.h(self.T)*mat.rho(self.T)*jac*dx(i)
        self.H_form = fem.form(f)
        self.Qloss_form = fem.form(alpha*(self.T - self.T_amb)*jac*ds(bc_idx))

        self.probes = Probe_writer(os.path.join(self.result_dir, 'probes.csv'))
        self.create_probes(self.probes)

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
        
        @probes.register_probe('T', '°C')
        def Tc_probe():
            return probe_function(self.T_probes_coords, self.domain, self.T)
        
        @probes.register_probe('power', 'W')
        def power():
            return self.q.value*self.Vc

    def create_solver(self, F, u):
        problem = fem.petsc.NonlinearProblem(F, u)
        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
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

    def solve_unsteady(self, Qc_t=None, T_amb_t=None):
        self.t = 0
        self.t_n = self.t
        self.T_amb.value = T_amb_t(self.t)
        self.q.value = Qc_t(self.t)/self.Vc
        t_start = time.time()
        next_xdmf_t = 0.0
        self.probes.evaluate_probes()
        self.probes.write_probes()
        self.probes.print()
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
                if max_T_diff > 0.25:
                    self.dt.value *= 0.9
                    self.dt.value = max(self.dt_min, self.dt.value)
                    continue
                elif max_T_diff < 0.1:
                    self.dt.value /= 0.95
                    self.dt.value = min(self.dt_max, self.dt.value)

                self.probes.evaluate_probes()
                success = True  

            while next_xdmf_t <= self.t:
                denom_dt = self.t - self.t_n
                enum_dt = next_xdmf_t - self.t_n
                self.T_xdmf.x.array[:] = self.T.x.array
                self.T_xdmf.x.array[:] -= self.T_n.x.array
                self.T_xdmf.x.array[:] *= enum_dt/denom_dt
                self.T_xdmf.x.array[:] += self.T_n.x.array
                self.xdmf.write_function(self.T_xdmf, next_xdmf_t)
                next_xdmf_t += self.dt_xdmf
            MPI.COMM_WORLD.Barrier()

            self.T_n.x.array[:] = self.T.x.array
            self.t_n = self.t

            if MPI.COMM_WORLD.rank == 0:
                self.elapsed = time.time() - t_start
                self.ielapsed = time.time() - i_start
            self.probes.print()
            self.probes.write_probes()

    def get_current_range(self, cell_tag=None):
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

    def plot(self):
        # TODO: move static stuff into offline phase, no need to calculate each time
        cells, types, x = plot.create_vtk_mesh(self.V)
        num_cells_local = self.domain.topology.index_map(self.domain.topology.dim).size_local
        num_dofs_local = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        num_dofs_per_cell = cells[0]
        cells_dofs = (np.arange(len(cells)) % (num_dofs_per_cell+1)) != 0
        global_dofs = self.V.dofmap.index_map.local_to_global(cells[cells_dofs].copy())
        cells[cells_dofs] = global_dofs

        root = 0
        global_cells = self.domain.comm.gather(cells[:(num_dofs_per_cell+1)*num_cells_local], root=root)
        global_types = self.domain.comm.gather(types[:num_cells_local])
        global_x = self.domain.comm.gather(x[:self.V.dofmap.index_map.size_local,:], root=root)
        global_vals = self.domain.comm.gather(self.T.x.array[:num_dofs_local], root=root)

        if MPI.COMM_WORLD.rank == 0:
            root_x = np.vstack(global_x)
            root_cells = np.concatenate(global_cells)
            root_types = np.concatenate(global_types)
            root_vals = np.concatenate(global_vals)

            grid = pyvista.UnstructuredGrid(root_cells, root_types, root_x)
            grid.point_data["u"] = root_vals
            grid.set_active_scalars("u")
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=False)
            plotter.view_xy()
            plotter.show()

    def pseudoexperimental_data_steady(self, Qc=100, T_amb=20):
        exp_data = Experiment_data()
        exp_data.steady_state_mean = self.solve_steady(Qc=Qc, T_amb=T_amb)
        exp_data.steady_state_mean['Power [W]'] = Qc
        exp_data.steady_state_mean['16 - Ambient [°C]'] = T_amb
        exp_data.steady_state_std = exp_data.steady_state_mean.copy()
        exp_data.steady_state_std.values[:] = 0.0
        exp_data.steady_state_mean.rename("Experiment Mean", inplace=True)
        exp_data.steady_state_std.rename("Experiment Std", inplace=True)

        return exp_data
        

# sim = Experiment(dim = 2)
# experiment_file = 'experiment_data/20231009_third/Test_TF24_Third_measurement_054411.csv'
# exp = Experiment_data(experiment_file)
# T_amb = exp.steady_state_mean['16 - Ambient [°C]']
# Qc = exp.steady_state_mean['Power [W]']
# comparer = SteadyStateComparer(sim, exp)
# exp.plot_data_series()
# exit()

# sim.mats[0].k.value[:] = np.array([20.89, 0.1875])
# sim.mats[1].k.value[:] = np.array([5.498e-02, 7.890e-06])
# sim.mats[2].k.value[:] = np.array([50.54, 0.0])
# sim.mats[3].k.value[:] = np.array([2.331, 0.0])
# sim.mats[3].k.value[:] = np.array([ 2.5052e+00, -6.4800e-04])
# sim.mats[4].k.value[:] = np.array([ 5.05460e-01,  8.49000e-05, -8.42500e-07, 1.96939e-09]) #1764
# comparer.update()
# d = [[5e-3, 5e-4],[1e-6, 1e-9],[5e-2, 5e-3],[1e-5, 1e-6],[1e-5, 1e-9, 1e-10, 1e-13]]


# Qc_t = lambda t: Qc
# T_amb_t = lambda t: T_amb
# sim.solve_unsteady(Qc_t=Qc_t, T_amb_t=T_amb_t)
# sim.close_results()
# m=4
# for j in range(300):
#     k = random.randint(0, len(d[m])-1)
#     sig = 1 if random.random() < 0.5 else -1
#     k = MPI.COMM_WORLD.bcast(k, root=0)
#     sig = MPI.COMM_WORLD.bcast(sig, root=0)
#     comparer.update()
#     for i in range(10):
#         local_d = d[m][k]
#         prev_e = comparer.total_square_error.copy()
#         sim.mats[m].k.value[k] += sig*local_d
#         try:
#             comparer.update()
#         except:
#             sim.mats[m].k.value[k] -= sig*local_d
#             break
#         new_e = comparer.total_square_error.copy()
#         if new_e >= prev_e:
#             sim.mats[m].k.value[k] -= sig*local_d
#             #comparer.update()
#             break
#         if MPI.COMM_WORLD.rank == 0:
#             abs_e = comparer.total_abs_error.copy()
#             sqr_e = comparer.total_square_error.copy()
#             print(abs_e, sqr_e, sim.mats[m].k.value, j, m)

# if MPI.COMM_WORLD.rank == 0:
#     print(sim.mats[0].k.value)
#     print(sim.mats[1].k.value)
#     print(sim.mats[2].k.value)
#     print(sim.mats[3].k.value)
#     print(sim.mats[4].k.value)

# comparer.print_data()
# comparer.save_data()
# comparer.create_figure(save_path=sim.result_dir + "/comparer.html")
# T_range = sim.get_current_range(cell_tag=1)
# sim.mats[0].plot(T_lim_used=T_range, save_path=sim.result_dir + "/mat0.html")
# T_range = sim.get_current_range(cell_tag=2)
# sim.mats[1].plot(T_lim_used=T_range, save_path=sim.result_dir + "/mat1.html")
# T_range = sim.get_current_range(cell_tag=3)
# sim.mats[2].plot(T_lim_used=T_range, save_path=sim.result_dir + "/mat2.html")
# T_range = sim.get_current_range(cell_tag=4)
# sim.mats[3].plot(T_lim_used=T_range, save_path=sim.result_dir + "/mat3.html")
# T_range = sim.get_current_range(cell_tag=5)
# sim.mats[4].plot(T_lim_used=T_range, save_path=sim.result_dir + "/mat4.html")
# sim.solve_steady(T_amb = T_amb, Qc=Qc)

