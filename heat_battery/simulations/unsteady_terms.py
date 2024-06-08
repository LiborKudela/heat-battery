from . import simulation_base as sb

class UnsteadyBoundaryTerm():
    def is_steady(self):
        return False
    
    def is_unsteady(self):
        return True
    
    def is_boundary(self):
        return True
    
    def is_volumetric(self):
        return False

class UnsteadyVolumetricTerm():
    def is_steady(self):
        return False
    
    def is_unsteady(self):
        return True
    
    def is_boundary(self):
        return False
    
    def is_volumetric(self):
        return True

class AmbientCooling(UnsteadyBoundaryTerm):
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.T_amb = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((20.0, 20.0)))
        self.alpha = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((1.0, 1.0)))
        self.T_amb_t = lambda t: 20.0
        self.alpha_t = lambda t: 1.0

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
    
    def __call__(self, T, t, x, boundary=None):
        return self._alpha(t)*(T - self._T_amb(t))

    def update(self, t):
        if t is self.sim.t:
            self.T_amb.value[0] = self.T_amb_t(t.value)
            self.alpha.value[0] = self.alpha_t(t.value)
        elif t is self.sim.t_n:
            self.T_amb.value[1] = self.T_amb_t(t.value)
            self.alpha.value[1] = self.alpha_t(t.value)

    def next_step(self):
        self.T_amb.value[1] = self.T_amb.value[0]
        self.alpha.value[1] = self.alpha.value[0]

class UniformHeatSource(UnsteadyVolumetricTerm):
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.Qc = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((1.0, 1.0)))
        self.Qc_t = lambda t: 1.0

    def Q_vol(self, t):
        # constrant heat per unit of volume
        if t is self.sim.t:
            qc = self.Qc[0]
        elif t is self.sim.t_n:
            qc = self.Qc[1]
        return qc
    
    def __call__(self, T, t, x, domain=None):
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.Q_vol(t)/Vc

    def update(self, t):
        if t is self.sim.t:
            self.Qc.value[0] = self.Qc_t(t.value)
        elif t is self.sim.t_n:
            self.Qc.value[1] = self.Qc_t(t.value)

    def next_step(self):
        self.Qc.value[1] = self.Qc.value[0]

class TemperatureControlledUniformHeatSource(UnsteadyVolumetricTerm):
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.Qc = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((1.0, 1.0)))
        self.Qc_gain = 0.0
        self.Qc_integral = 0.0
        self.Qc_derivative = 0.0
        self.Qc_start = 100
        self.T_eval = None
        self.T_limit = None
        self.power_limit = None
        self.kg = 100
        self.kI = 0.01
        self.kd = 100.0
        self.prev_diff = 380

    def Q_vol(self, t):
        # constrant heat per unit of volume
        if t is self.sim.t:
            qc = self.Qc[0]
        elif t is self.sim.t_n:
            qc = self.Qc[1]
        return qc
    
    def __call__(self, T, t, x, domain=None):
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.Q_vol(t)/Vc

    def update(self, t):
        if t is self.sim.t:
            diff = self.T_limit - self.T_eval()
            self.Qc_gain = self.kg*diff
            self.Qc_integral = self.Qc_integral + self.kI*diff*(self.sim.t.value-self.sim.t_n.value)
            self.Qc_derivative = self.kd*(diff-self.prev_diff)/(self.sim.t.value-self.sim.t_n.value)
            self.Qc.value[0] = self.Qc_gain + self.Qc_integral + self.Qc_derivative
            self.Qc.value[0] = min(self.power_limit, self.Qc.value[0])
            self.prev_diff = diff
        elif t is self.sim.t_n:
            self.Qc.value[1] = self.Qc_start

    def next_step(self):
        self.Qc.value[1] = self.Qc.value[0]

class SerialWiresResistiveHeating(UnsteadyVolumetricTerm):
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.current = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((0.5, 0.5))) # can change value w/o recompilation (v, v_n)
        
    def dR_dx(self, T, domain):
        #dudx = I/A*r(T(x))
        wire_A = sb.np.pi*self.sim.geo_meta['call_data']['custom_data'][domain]['diameter']**2/4
        return 1/wire_A**2*self.sim.mats[domain].sigma(T)
    
    def q_vol(self, T, t, domain):
        #Q = I*U
        if t is self.sim.t:
            I = self.current[0]
        elif t is self.sim.t_n:
            I = self.current[1]
        return I**2*self.dR_dx(T, domain)
    
    def __call__(self, T, t, x, domain=None):
        return self.q_vol(T, t, domain)

    def update(self, t):
        pass

    def next_step(self):
        self.current.value[1] = self.current.value[0]