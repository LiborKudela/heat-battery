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
        self.new_constant('Qc', 0.0)
        self.set_update('Qc', lambda t: 0.0)
        self.new_integral('Q_cumulative', 0.0, 'Qc')
    
    def __call__(self, T, x, t=None, domain=None):
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.get_constant('Qc', t)/Vc
    
class TemperatureLimitedUniformHeatSource(Term):

    def define_terms(self):
        self.T_limit = 100.0
        self.T_limit_tol = 0.1
        self.toggle_sens = 0.0001   

        self.new_constant("toggle", 1.0)
        self.set_update('toggle', self.toggle_update)

        self.new_constant("Q_in", 0.0)
        self.set_update('Q_in', lambda t: 0.0)
        self.new_integral('Q_in_cumulative', 0.0, 'Q_in')

        self.new_constant("T_probe", 0.0)
        self.set_update('T_probe', lambda t: 0.0)

        self.new_constant('Qc', 0.0)
        self.set_update('Qc', self.Qc_update, prevent_override=True)
        self.new_integral('Qc_cumulative', 0.0, 'Qc')

    def set_T_limit(self, value):
        self.T_limit = value

    def set_sens(self, value):
        self.toggle_sens = value

    def toggle_update(self, t):
        T_probe = self.fem_consts['T_probe'].value[0]
        toggle_n = self.fem_consts['toggle'].value[1]
        diff = self.T_limit - T_probe
        diff = self.sim.dt.value*self.T_limit - self.get_integral_step('T_probe')
        return sb.np.clip(toggle_n+diff*self.toggle_sens, 0, 1)
    
    def Qc_update(self, t):
        toggle = self.fem_consts['toggle'].value[0]
        Q_in = self.fem_consts['Q_in'].value[0]
        return Q_in*toggle
    
    def __call__(self, T, x, t=None, domain=None):
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.get_constant('Qc', t)/Vc

class PIDControlledHeatSource(PIDTerm):
    """This is a typical PID term that is used to control power based on
    temperature probe."""

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