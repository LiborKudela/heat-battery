from heat_battery.simulations.simulation_base import MPI, Simulation, fem
from heat_battery.simulations.probing import FunctionSampler
from heat_battery.simulations import terms

class Experiment_v1(Simulation):
    def define_form_subdomain_terms(self):

        # add dynamic source term for malapa cartridge
        self.set_source_term(terms.UniformHeatSource(self), "heated cartridge")

        # add dynamic boundary term for ambient cooling on the surface
        self.set_bc_term(terms.AmbientCooling(self), "outer_surface")

    def create_common_probes(self, probes):
        self.T_probes_coords = list(self.geo_meta["points"]["T"].values())
        self.T_probes_names = list(self.geo_meta["points"]["T"].keys())
        sampler = FunctionSampler(self.T_probes_coords, self.domain)

        @probes.register_probe("T", unit="°C")
        def Tc_probe():
            return sampler.eval(self.T)

        # form for calculating heat in the whole domain
        h_form = 0.0
        for i, mat in enumerate(self.mats, 1):
            h_form += mat.h(self.T)*mat.rho(self.T)*self.jac*self.dx(i)
        H_form = fem.form(h_form)

        @probes.register_probe("heat", "J")
        def H_probe():
            H = fem.assemble_scalar(H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H

    def solve_steady(self, Qc=10, T_amb=20, alpha=6.3, **kwargs):
        obj = self.get_source_term("heated cartridge")
        obj.set_steady_state_value('Qc', Qc)

        obj = self.get_bc_term("outer_surface")
        obj.set_steady_state_value('T_amb', T_amb)
        obj.set_steady_state_value('alpha', alpha)
        probes = super().solve_steady(**kwargs)
        return probes

    def solve_unsteady(
            self, 
            Qc_t=None, 
            T_amb_t=None, 
            alpha_t=None, 
            **kwargs
        ):
        if Qc_t is not None:
            obj: terms.UniformHeatSource
            obj = self.get_source_term("heated cartridge")
            obj.set_update('Qc', Qc_t)
        if T_amb_t is not None:
            obj: terms.AmbientCooling
            obj = self.get_bc_term("outer_surface")
            obj.set_update('T_amb', T_amb_t)
            obj.set_update('alpha', alpha_t)
        return super().solve_unsteady(**kwargs)
