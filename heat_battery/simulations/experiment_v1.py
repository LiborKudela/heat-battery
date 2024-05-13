from . import steady_terms, unsteady_terms
from .probing import FunctionSampler
from .simulation_base import MPI, Simulation, fem, pd


class Experiment_v1(Simulation):
    def define_form_subdomain_terms(self):

        # add dynamic source term for malapa cartridge
        self.set_unsteady_source_term(unsteady_terms.UniformHeatSource(self), "heated cartridge")

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

        @probes.register_probe("heat", "J")
        def H_probe():
            H = fem.assemble_scalar(H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H

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

    def solve_unsteady(self, Qc_t=None, T_amb_t=None, alpha_t=None, **kwargs):
        if Qc_t is not None:
            obj = self.get_unsteady_source_term("heated cartridge")
            obj.Qc_t = Qc_t
        if T_amb_t is not None:
            obj = self.get_unsteady_bc_term("outer_surface")
            obj.T_amb_t = T_amb_t
            obj.alpha_t = alpha_t
        return super().solve_unsteady(**kwargs)
