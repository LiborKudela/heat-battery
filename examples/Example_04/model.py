from heat_battery.simulations.simulation_base import MPI, Simulation, fem, pd, ufl
from heat_battery.simulations.probing import FunctionSampler
from heat_battery.simulations import terms

class PassiveStorage(Simulation):
    def define_form_subdomain_terms(self):

        # add dynamic source term for malapa cartridge
        self.set_source_term(terms.PIDControlledHeatSource(self), "heated cartridge")

        # add dynamic boundary term for ambient cooling on the outer surface
        self.set_bc_term(terms.AmbientCooling(self), "outer_surface")

        # add dynamic boundary term for ambient cooling on the tubes surface
        self.set_bc_term(terms.AmbientCooling(self), "mebrane_surface")

    def create_unsteady_probes(self, probes):
        super().create_unsteady_probes(probes)

        self.T_probes_coords = list(self.geo_meta["points"]["T"].values())
        self.T_probes_names = list(self.geo_meta["points"]["T"].keys())
        sampler = FunctionSampler(self.T_probes_coords, self.domain)

        @probes.register_probe("T", "°C")
        def Tc_probe():
            return sampler.eval(self.T)

        # form for calculating heat in the whole domain
        h_form = 0.0
        for i, mat in enumerate(self.mats, 1):
            h_form += mat.h(self.T)*mat.rho(self.T)*self.jac*self.dx(i)
        H_form = fem.form(h_form)

        @probes.register_probe("heat", "kWh")
        def H_probe():
            H = fem.assemble_scalar(H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H*2.77777778e-7
        
        bc_obj = self.get_bc_term('outer_surface')
        qloss_form = fem.form(bc_obj(self.T, self.x, self.t)*self.jac*self.get_measure_ds('outer_surface'))
        @probes.register_probe('heat_loss', 'W')
        def loss_probe():
            q_flow = fem.assemble_scalar(qloss_form)
            q_flow = self.domain.comm.allreduce((q_flow), op=MPI.SUM)
            return q_flow
        
        mem_obj = self.get_bc_term('mebrane_surface')
        mem_qloss_form = fem.form(mem_obj(self.T, self.x, self.t)*self.jac*self.get_measure_ds('mebrane_surface'))
        @probes.register_probe('heat_loss_mem', 'W')
        def loss_probe():
            q_flow = fem.assemble_scalar(mem_qloss_form)
            q_flow = self.domain.comm.allreduce((q_flow), op=MPI.SUM)
            return q_flow
        
        obj = self.get_source_term('heated cartridge')
        power_form = fem.form(obj(self.T, self.x, self.t, 'heated cartridge')*self.jac*self.get_measure_dx('heated cartridge'))
        @probes.register_probe('power', 'W')
        def power():
            q_power = fem.assemble_scalar(power_form)
            q_power = self.domain.comm.allreduce((q_power), op=MPI.SUM)
            return q_power
        
        area_surf = fem.assemble_scalar(fem.form(self.jac*self.get_measure_ds('outer_surface')))
        area_surf = self.domain.comm.allreduce((area_surf), op=MPI.SUM)
        Tmean_surf = fem.form(self.T/area_surf*self.jac*self.get_measure_ds('outer_surface'))
        @probes.register_probe('Ts_avg', 'T')
        def average_temp():
            value = fem.assemble_scalar(Tmean_surf)
            value = self.domain.comm.allreduce((value), op=MPI.SUM)
            return value
        
        area_mem = fem.assemble_scalar(fem.form(self.jac*self.get_measure_ds('mebrane_surface')))
        area_mem = self.domain.comm.allreduce((area_mem), op=MPI.SUM)
        Tmean_mem = fem.form(self.T/area_mem*self.jac*self.get_measure_ds('outer_surface'))
        @probes.register_probe('Tm_avg', 'T')
        def average_temp():
            value = fem.assemble_scalar(Tmean_mem)
            value = self.domain.comm.allreduce((value), op=MPI.SUM)
            return value
        
        #TODO: This is temporary fix when using dS measures for https://fenicsproject.discourse.group/t/scallar-assembly-of-a-internal-surface-integral-misbehaves-in-parallel/14655/3
        #      remove this for loop when PR from @dokken is merged and backported to 0.8
        for i in range(self.domain.topology.dim + 1):
            self.domain.topology.create_entities(i)

        area_cartridge = fem.assemble_scalar(fem.form(8*self.get_measure_dS('cartridge_surface')))
        area_cartridge = self.domain.comm.allreduce((area_cartridge), op=MPI.SUM)
        Tmean_cartridge = fem.form(self.T/area_cartridge*self.jac*self.get_measure_dS('cartridge_surface'))
        @probes.register_probe('Tc_avg', 'T')
        def average_temp():
            value = fem.assemble_scalar(Tmean_cartridge)
            value = self.domain.comm.allreduce((value), op=MPI.SUM)
            return value
        
        pid_obj = self.get_source_term('heated cartridge')
        @probes.register_probe('PID', '-')
        def pid_vals():
            return pid_obj.get_current_pid_values()

    def solve_steady(self, Qc=10, T_amb=20, xdmf_file=None, alpha=6.3):
        #FIXME: This will not work for currente version of PID type Term
        obj = self.get_source_term("heated cartridge")
        obj.Qc.value = Qc

        obj = self.get_bc_term("outer_surface")
        obj.T_amb.value = T_amb
        obj.alpha.value = alpha
        super().solve_steady(xdmf_file=xdmf_file)

        return pd.Series(
            data=self.probes.get_value("T"),
            index=self.T_probes_names,
            name="Simulation",
        )

    def solve_unsteady(self, 
        T_pid_input_control=lambda t: 400.0,
        T_amb_t=lambda t: 18.0,
        alpha_t=lambda t: 2.5,
        alpha_mem_t=lambda t: 1.0, 
        pid=(100.0, 0.02, 100.0),
        pid_ctrl_interval=(10.0, 30.0),
        pid_power_lims = [0, float('Inf')],
        **kwargs):

        obj = self.get_source_term("heated cartridge")
        obj.set_pid(*pid)
        obj.set_converge_tol(1e-2)
        obj.set_output_limits(pid_power_lims)
        obj.set_ctrl_interval(pid_ctrl_interval)
        obj.set_reference(T_pid_input_control)
        obj.set_probe(lambda t: self.unsteady_probes.get_value('Tc_avg'))

        obj = self.get_bc_term("outer_surface")
        obj.set_update('T_amb', T_amb_t)
        obj.set_update('alpha', alpha_t)
        
        obj = self.get_bc_term("mebrane_surface")
        obj.set_update('T_amb', T_amb_t)
        obj.set_update('alpha', alpha_mem_t)

        return super().solve_unsteady(**kwargs)