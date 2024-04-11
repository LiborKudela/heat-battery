from .simulation_base import Simulation, fem, PETSc, pd, FunctionSampler, MPI

class Expression_Qc_unsteady:
    def __init__(self, sim):
        self.sim = sim
        self.Qc = fem.Constant(self.sim.domain, PETSc.ScalarType((1.0, 1.0)))
        i = self.sim.subdomain_map['heated cartridge']
        self.Vc = self.sim.V_subdomain[i]
        self.Qc_t = None
        self.alpha_t = None

    def q_vol(self, t):
        # constrant heat per unit of volume
        if t is self.sim.t:
            qc = self.Qc[0]/self.Vc
        elif t is self.sim.t_n:
            qc = self.Qc[1]/self.Vc
        return qc
    
    def __call__(self, T, t, x):
        return self.q_vol(t)

    def update(self, t):
        if t is self.sim.t:
            self.Qc.value[0] = self.Qc_t(t.value)
        elif t is self.sim.t_n:
            self.Qc.value[1] = self.Qc_t(t.value)

    def next_step(self):
        self.Qc.value[0] = self.Qc.value[1]

class Expression_Tamb_unsteady:
    def __init__(self, sim):
        self.sim = sim
        self.T_amb = fem.Constant(self.sim.domain, PETSc.ScalarType((1.0, 1.0)))
        self.alpha = fem.Constant(self.sim.domain, PETSc.ScalarType((6.3, 6.3)))
        self.T_amb_t = None
        self.alpha_t = None

    def _T_amb(self, t):
        # constrant heat per unit of volume
        if t is self.sim.t:
            T_amb = self.T_amb[0]
        elif t is self.sim.t_n:
            T_amb = self.T_amb[1]
        return T_amb
    
    def _alpha(self, t):
        # constrant heat per unit of volume
        if t is self.sim.t:
            alpha = self.alpha[0]
        elif t is self.sim.t_n:
            alpha = self.alpha[1]
        return alpha
    
    def __call__(self, T, t, x):
        return self._alpha(t)*(T - self._T_amb(t))

    def update(self, t):
        if t is self.sim.t:
            self.T_amb.value[0] = self.T_amb_t(t.value)
            self.alpha.value[0] = self.alpha_t(t.value)
        elif t is self.sim.t_n:
            self.T_amb.value[1] = self.T_amb_t(t.value)
            self.alpha.value[1] = self.alpha_t(t.value)

    def next_step(self):
        self.T_amb.value[0] = self.T_amb.value[1]
        self.alpha.value[0] = self.alpha.value[1]

class Expression_Qc_steady_state:
    def __init__(self, sim):
        self.sim = sim
        self.Qc = fem.Constant(self.sim.domain, PETSc.ScalarType((10.0)))
        i = self.sim.subdomain_map['heated cartridge']
        self.Vc = self.sim.V_subdomain[i]

    def q_vol(self):
        # constrant heat per unit of volume
        qc = self.Qc/self.Vc
        return qc
    
    def __call__(self, T, x):
        return self.q_vol()

    def update(self):
        pass

class Expression_Tamb_steady_state:
    def __init__(self, sim):
        self.sim = sim
        self.T_amb = fem.Constant(self.sim.domain, PETSc.ScalarType((20.0)))
        self.alpha = fem.Constant(self.sim.domain, PETSc.ScalarType(6.3))
    
    def __call__(self, T, x):
        # T, x must be in the signature coz the form constructor will insert them
        # you do not need to use them if not needed
        return self.alpha*(T - self.T_amb)

    def update(self):
        self.T_amb.value = 20

class Experiment_v1(Simulation):
    def define_form_subdomain_terms(self):
        self.set_unsteady_source_term(Expression_Qc_unsteady, 'heated cartridge')
        self.set_steady_state_source_term(Expression_Qc_steady_state, 'heated cartridge')
        self.set_unsteady_bc_term(Expression_Tamb_unsteady, 'outer_surface')
        self.set_steady_state_bc_term(Expression_Tamb_steady_state, 'outer_surface')

    def create_probes(self, probes):
        super().create_probes(probes)

        self.T_probes_coords = list(self.geo_meta['probes']['T'].values())
        self.T_probes_names = list(self.geo_meta['probes']['T'].keys())
        sampler = FunctionSampler(self.T_probes_coords, self.domain)
        @probes.register_probe('T', '°C')
        def Tc_probe():
            return sampler.eval(self.T)
        
        # form for calculating heat in the whole domain
        h_form = 0
        for i, mat in enumerate(self.mats, 1):
            h_form += mat.h(self.T)*mat.rho(self.T)*self.jac*self.dx(i)
        H_form = fem.form(h_form)
        @probes.register_probe('heat', 'J')
        def H_probe():
            H = fem.assemble_scalar(H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H

    def solve_steady(self, Qc=10, T_amb=20, save_xdmf=False, alpha=6.3):
        obj = self.get_steady_steady_source_term('heated cartridge')
        obj.Qc.value = Qc

        obj = self.get_steady_state_bc_term('outer_surface')
        obj.T_amb.value = T_amb
        obj.alpha.value = alpha
        super().solve_steady(save_xdmf=save_xdmf)

        return pd.Series(data=self.probes.get_value('T'), index=self.T_probes_names, name="Simulation")
    
    def solve_unsteady(self, Qc_t=None, T_amb_t=None, alpha_t=None, **kwargs):
        if Qc_t is not None:
            obj = self.get_unsteady_source_term('heated cartridge')
            obj.Qc_t = Qc_t
        if T_amb_t is not None:
            obj = self.get_unsteady_bc_term('outer_surface')
            obj.T_amb_t = T_amb_t
            obj.alpha_t = alpha_t
        return super().solve_unsteady(**kwargs)
