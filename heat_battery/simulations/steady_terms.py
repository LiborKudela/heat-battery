from . import simulation_base as sb

class SteadyBoundaryTerm():
    def is_steady(self):
        return True
    
    def is_unsteady(self):
        return False
    
    def is_boundary(self):
        return True
    
    def is_volumetric(self):
        return False

class SteadyVolumetricTerm():
    def is_steady(self):
        return True
    
    def is_unsteady(self):
        return False
    
    def is_boundary(self):
        return False
    
    def is_volumetric(self):
        return True

class AmbientCooling(SteadyBoundaryTerm):
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.T_amb = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((20.0)))
        self.alpha = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType(6.3))
    
    def __call__(self, T, x, boundary=None):
        return self.alpha*(T - self.T_amb)
    
class UniformHeatSource(SteadyVolumetricTerm):
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.Qc = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((10.0)))
    
    def __call__(self, T, x, domain=None):
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.Qc/Vc
    
class SerialWiresResistiveHeating(SteadyVolumetricTerm):
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.current = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((0.1)))
        self.domain_calls = []
        
    def dR_dx(self, T, domain):
        #dudx = I/A*r(T(x))
        wire_A = sb.np.pi*self.sim.geo_meta['call_data']['custom_data'][domain]['diameter']**2/4
        return 1/wire_A**2*self.sim.mats[domain].sigma(T)
    
    def q_vol(self, T, domain):
        #Q = I*U
        I = self.current 
        return I**2*self.dR_dx(T, domain)
    
    def __call__(self, T, x, domain=None):
        self.domain_calls.append(domain)
        return self.q_vol(T, domain)

    def update(self):
        pass
