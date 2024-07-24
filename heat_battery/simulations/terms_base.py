from . import simulation_base as sb

class Term():
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.fem_consts = {}
        self.fem_const_prevent_override = {}
        self.fem_consts_steady = {}
        self.update_functions = {}
        self.define_terms()

    def __call__(self, T, x, t=None, domain=None):
        "This is used to "
        pass

    def define_terms(self):
        """This function is called upon instantiation of the Term and is used to
        define all internals of the Term (such as fem_consts and their update
        callbacks)"""
        pass
      
    def new_constant(self, name, value):
        """Defines new fem.constant in the term that has initial 'value'.
           'no_update' specifies if it should have set update callback.
           Warrning will be produces if inconsistency is detected.
        """
        self.fem_consts[name] = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((value, value)))
        self.fem_consts_steady[name] = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType(value))
    
    def set_update(self, name, f, prevent_override=False):
        """Sets update callback 'f' for constant 'name'. Future overrides can 
        be prevented with 'prevent_override'."""
        if self.fem_const_prevent_override.get(name):
            raise Exception(f"Constant {name} update has set 'prevent_override'")
        self.update_functions[name] = f
        self.fem_const_prevent_override[name] = prevent_override
    
    def get_constant(self, name, t):
        """Returns appropriate ulf form value of constant. This is usually used
        in '__call__' method to define ufl form of the Term. When 't' is 'None'
        this will give ufl form value for steady state formulations."""
        if t is not None:
            return self.fem_consts[name][self.get_time_index(t)]
        else:
            return self.fem_consts_steady[name]
    
    def set_initial_value(self, name, value):
        """Sets initial value of constant 'name' for the Term (perhaps dynamic)
        for the begining of unsteady simulations."""
        #TODO: add assert to check that this is not beeing used after initialisation
        self.fem_consts[name].value[:] = value

    def set_steady_state_value(self, name, value):
        """Sets 'value' of constant 'name' of the Term for the steady-state
        simulations."""
        self.fem_consts_steady[name].value = value

    def warn_constants_with_no_update(self):
        """Produces warnings for constants that are declared with no updates 
        callbacks in Term. It is good practice to call this for each Term right
        before simulation"""
        for name, fc in self.fem_consts.items():
            if self.update_functions.get(name) is None:
                print(f"Warning: FEM Constant {name} has no coresponding update function")

    def get_time_index(self, t):
        """Resolves what part of Crank-Nicolson scheme is beeing dealt with.
           Steady state form is requested by the fact that type(t)==None"""
        if t is self.sim.t:
            return 0
        elif t is self.sim.t_n:
            return 1
        else:
            raise Exception(f"Unexpected time arg: {t} of type {type(t)}")
        
    def update(self, t):
        #FIXME: TERM - This needs steady-state update as well
        """When simulation changes dt, this sets new values in it."""
        i = self.get_time_index(t)
        for name, fc in self.fem_consts.items():
            update_function = self.update_functions.get(name)
            if update_function is not None:
                val = self.update_functions[name](t.value)
                assert val is not None, f"Update callback for '{name}' returned None"
                fc.value[i] = val
        
    def next_step(self):
        """After time step is finished, it is called internaly by unsteady
        simulation loop to roll last values into previous time step for the 
        consistency of Cranck-Nicolson Scheme."""
        for name, fc in self.fem_consts.items():
            fc.value[1] = fc.value[0]

class PIDTerm(Term):
    
    def define_terms(self):

        self.k_p = 1.0
        self.k_i = 0.0
        self.k_d = 0.0

        # update diference first
        self.new_constant('diff', 0.0)
        self.set_update('diff', self.diff_update, prevent_override=True)

        # update all internal (prevent_override=True) stuff
        self.new_constant('p', 1.0)
        self.set_update('p', self.p_update, prevent_override=True)
        self.new_constant('i', 1.0)
        self.set_update('i', self.i_update, prevent_override=True)
        self.new_constant('d', 1.0)
        self.set_update('d', self.d_update, prevent_override=True)

        # at last update the output value
        self.new_constant('output', 0.0)
        self.set_update('output', self.out_update, prevent_override=True)

    def set_pid(self, p, i, d):
        self.k_p = p
        self.k_i = i
        self.k_d = d

    def set_probe(self, f):
        #FIXME: PID TERM - This should be constant so there is steady equivalent automaticaly
        self.probe_f = f

    def set_reference(self, f):
        #FIXME: PID TERM - This should be constant so there is steady equivalent automaticaly
        self.ref_f = f

    def diff_update(self, t):
        ref = self.ref_f(t)
        rel_val = self.probe_f(t)
        return ref - rel_val
    
    def p_update(self, t):
        diff = self.fem_consts['diff'].value[0]
        return self.k_p*diff
    
    def i_update(self, t):
        diff = self.fem_consts['diff'].value[0]
        prev_i = self.fem_consts['i'].value[1]
        return prev_i + self.k_i*diff*(self.sim.dt.value)
    
    def d_update(self, t):
        diff = self.fem_consts['diff'].value[0]
        diff_n = self.fem_consts['diff'].value[1]
        return self.k_d*(diff-diff_n)/(self.sim.dt.value)

    def out_update(self, t):
        """Return form that represent PID controled value"""
        p = self.fem_consts['p'].value[0]
        i = self.fem_consts['i'].value[0]
        d = self.fem_consts['d'].value[0]
        return p + i + d
    
    def __call__(self, T, x, t=None, domain=None):
        """Return ufl form that represents the output. This should be overriden
        in subclasses."""
        return self.get_constant('output', t)
    
