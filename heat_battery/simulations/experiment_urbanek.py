from . import steady_terms, unsteady_terms
from .probing import FunctionSampler
from .simulation_base import MPI, Simulation, fem, pd, ufl


class Experiment_urbanek(Simulation):
    def define_form_subdomain_terms(self):

        # add dynamic source term for malapa cartridge
        self.set_unsteady_source_term(unsteady_terms.TemperatureControlledUniformHeatSource(self), "heated cartridge")

        # add dynamic boundary term for ambient cooling on the surface
        self.set_unsteady_bc_term(unsteady_terms.AmbientCooling(self), "outer_surface")

        # add steady source term for malapa cartridge
        self.set_steady_state_source_term(steady_terms.UniformHeatSource(self), "heated cartridge")

        # add dynamic boundary term for ambient cooling on the surface
        self.set_steady_state_bc_term(steady_terms.AmbientCooling(self), "outer_surface")

    def create_probes(self, probes):
        super().create_probes(probes)

        self.T_probes_coords = list(self.geo_meta["probes"]["T"].values())
        self.T_probes_names = list(self.geo_meta["probes"]["T"].keys())
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
        
        bc_obj = self.get_unsteady_bc_term('outer_surface')
        qloss_form = fem.form(bc_obj(self.T, self.t, self.x)*self.jac*self.get_measure_ds('outer_surface'))
        @probes.register_probe('heat_loss', 'W')
        def loss_probe():
            q_flow = fem.assemble_scalar(qloss_form)
            q_flow = self.domain.comm.allreduce((q_flow), op=MPI.SUM)
            return q_flow
        
        obj = self.get_unsteady_source_term('heated cartridge')
        power_form = fem.form(obj(self.T, self.t, self.x, 'heated cartridge')*self.jac*self.get_measure_dx('heated cartridge'))
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

    def solve_steady(self, Qc=10, T_amb=20, save_xdmf=False, alpha=6.3):
        obj = self.get_steady_steady_source_term("heated cartridge")
        obj.Qc.value = Qc

        obj = self.get_steady_state_bc_term("outer_surface")
        obj.T_amb.value = T_amb
        obj.alpha.value = alpha
        super().solve_steady(save_xdmf=save_xdmf)

        return pd.Series(
            data=self.probes.get_value("T"),
            index=self.T_probes_names,
            name="Simulation",
        )

    def solve_unsteady(self, T_amb_t=None, alpha_t=None, T_cartridge_limit=None, power_limit=20000, **kwargs):
        if T_cartridge_limit is not None:
            obj = self.get_unsteady_source_term("heated cartridge")
            obj.T_limit = T_cartridge_limit
            obj.T_eval = lambda: self.probes.get_value('Tc_avg')
            obj.power_limit = power_limit
        if T_amb_t is not None:
            obj = self.get_unsteady_bc_term("outer_surface")
            obj.T_amb_t = T_amb_t
            obj.alpha_t = alpha_t
        return super().solve_unsteady(**kwargs)
