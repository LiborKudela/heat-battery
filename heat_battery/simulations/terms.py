from .terms_base import Term, PIDTerm, sb

class AmbientCooling(Term):

    def define_terms(self):
        self.new_constant('T_amb', 20.0)
        self.set_update('T_amb', lambda t: 20.0)

        self.new_constant('alpha', 6.3)
        self.set_update('alpha', lambda t: 6.3)
        
    def __call__(self, T, x, t=None, domain=None):
        return self.get_constant('alpha', t)*(T - self.get_constant('T_amb', t))

class UniformHeatSource(Term):

    def define_terms(self):
        self.new_constant('Qc', 1.0)
        self.set_update('Qc', lambda t: 1.0)
    
    def __call__(self, T, x, t=None, domain=None):
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.get_constant('Qc', t)/Vc

class PIDControlledHeatSource(PIDTerm):
    def __call__(self, T, x, t=None, domain=None):
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.get_constant('output', t)/Vc
    
    def converged(self):
        d_diff  = self.fem_consts['diff'].value[0] - self.diff_update(self.sim.t.value)
        return abs(d_diff) < 1e-2

class SerialWiresResistiveHeating(Term):
    
    def define_terms(self):
        self.new_constant('current', 0.5)
        self.set_update('current', lambda t: 0.5)
        
    def dR_dx(self, T, domain):
        #dudx = I/A*r(T(x))
        wire_A = sb.np.pi*self.sim.get_custom_data(domain)['diameter']**2/4
        return 1/wire_A**2*self.sim.mats[domain].sigma(T)
    
    def __call__(self, T, x, t=None, domain=None):
        assert domain is not None, "This Term must have domain kwarg set."
        return self.get_constant('current', t)**2*self.dR_dx(T, domain)