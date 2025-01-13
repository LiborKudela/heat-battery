from . import simulation_base as sb

class Term():
    def __init__(self, sim: sb.Simulation):
        self.sim = sim
        self.fem_consts = {}
        self.fem_const_prevent_override = {}
        self.fem_consts_steady = {}
        self.update_functions = {}
        #self.integral_states = []
        self.define_terms()

    def __call__(self, T, x, t=None, domain=None):
        "This is used to "
        pass

    def define_terms(self):
        """This function is called upon instantiation of the Term and is used to
        define all internals of the Term (such as fem_consts and their update
        callbacks)

        This function should be overridden in each Term class.
        """
        pass
      
    def new_constant(self, name, value):
        """Defines new fem.constant in the term.
           Warrning will be produces if there is already a constant with the same name.

        Args:
            name (str): Name of the constant
            value (float): Initial value of the constant
        """
        assert (self.fem_consts.get(name) is None), (
            f"Constant with name {name} already declared. Chose different name"
        )

        self.fem_consts[name] = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((value, value)))
        self.fem_consts_steady[name] = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType(value))

    def new_integral(self, name, value, parrent):
        """Defines new fem.constant in the term that is used to store integral
        of the parrent constant.

        Args:
            name (str): Name of the constant
            value (float): Initial value of the constant
            parrent (str): Name of the constant that is being integrated
        """
        self.fem_consts[name] = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType((value, value)))
        self.fem_consts_steady[name] = sb.fem.Constant(self.sim.domain, sb.PETSc.ScalarType(value))
        def integral_updater(t):
            return self.fem_consts[name].value[1] + self.get_integral_step(parrent)
        self.set_update(name, integral_updater, prevent_override=True)

    def update_from_form_integral(self, domain):
        """Updates integral constant from form expression.

        Args:
            domain (str): Name of the domain for which the update is being done
        """
        expression = self.__call__(self.sim.T, self.sim.x, self.sim.t, domain=domain)
        form = sb.fem.form(expression*self.sim.jac*self.sim.get_measure_ds(domain))
        def update_callback(t):
            local_value = sb.fem.assemble_scalar(form)
            global_value = self.sim.domain.comm.allreduce((local_value), op=sb.MPI.SUM)
            return global_value
        return update_callback
    
    def set_update(self, name, f, prevent_override=False):
        """Sets update callback 'f' for constant 'name'. Future overrides of 
        this callback can be prevented with 'prevent_override'.

        Args:
            name (str): Name of the constant for which update is being set
            f(t) (function): Update callback function of time 't' that returns new value at each call
            prevent_override (bool): If True, future overrides of this callback will raise an exception
        """
        if self.fem_consts.get(name) is None:
            raise Exception(f'Constant {name} does not exist')
        if self.fem_const_prevent_override.get(name):
            raise Exception(f"Constant {name} update has set 'prevent_override'")
        self.update_functions[name] = f
        self.fem_const_prevent_override[name] = prevent_override
    
    def get_constant(self, name, t):
        """Returns appropriate ulf form value of constant. This is usually used
        in '__call__' method to define ufl form of the Term. When 't' is 'None'
        this will give ufl form value for steady state formulations.

        Args:
            name (str): Name of the constant
            t (float): Time at which the constant is being evaluated
        """
        if t is not None:
            return self.fem_consts[name][self.get_time_index(t)]
        else:
            return self.fem_consts_steady[name]

    def get_current_value(self, name):
        """Returns current value of constant 'name'.

        Args:
            name (str): Name of the constant
        """
        return self.fem_consts[name].value[0]
    
    def get_current_values(self, names):
        """Returns current values of constants 'names'.

        Args:
            names (list): List of names of the constants
        """
        values = []
        for name in names:
            values.append(self.get_current_value(name))
        return values
    
    def get_integral_step(self, name):
        """Returns integral increment of constant 'name'.

        Args:
            name (str): Name of the constant
        """
        v = self.sim.theta*(self.sim.dt.value)*self.fem_consts[name].value[0]
        v_n = self.sim.theta*(self.sim.dt.value)*self.fem_consts[name].value[1]
        return v + v_n
    
    def set_initial_value(self, name, value):
        """Sets initial value of constant 'name' for the Term (perhaps dynamic)
        for the begining of unsteady simulations.

        Args:
            name (str): Name of the constant
            value (float): Initial value of the constant
        """
        #TODO: add assert to check that this is not beeing used after initialisation
        self.fem_consts[name].value[:] = value

    def set_steady_state_value(self, name, value):
        """Sets 'value' of constant 'name' of the Term for the steady-state
        simulations.

        Args:
            name (str): Name of the constant
            value (float): Value of the constant
        """
        self.fem_consts_steady[name].value = value

    def warn_constants_with_no_update(self):
        """Produces warnings for constants that are declared with no updates 
        callbacks in Term. It is good practice to call this for each Term right
        before simulation"""
        for name, fc in self.fem_consts.items():
            if self.update_functions.get(name) is None:
                print(f"Warning: FEM Constant {name} has no coresponding update function")

    def get_time_index(self, t):
        """Resolves what part of Crank-Nicolson scheme is beeing dealt with by
        checking if 't' is the same as 'self.sim.t' (which is time for the k+1
        step) or 'self.sim.t_n' (which is time for the k step). Steady state form
        is requested by the fact that type(t)==None

        Args:
            t (float): Time at which the constant is being evaluated
        """
        if t is self.sim.t:
            return 0
        elif t is self.sim.t_n:
            return 1
        else:
            raise Exception(
                f"Unexpected time arg: {t} of type {type(t)}"
                "t must be a fem.Constant from simulation class"
            )
        
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

    def get_checkpoint_data(self):
        """Returns data that is needed to checkpoint the Term. This is used by
        the simulation to save the state of the Term for restart."""
        data = {'unsteady': {}, 'steady': {}}
        for name, fc in self.fem_consts.items():
            data['unsteady'][name] = self.fem_consts[name].value
            data['steady'][name] = self.fem_consts_steady[name].value
        return data

    def load_checkpoint_data(self, data):
        """Loads data that is needed to checkpoint the Term. This is used by
        the simulation to load the state of the Term for restart."""
        for name, fc in self.fem_consts.items():
            self.fem_consts[name].value = data['unsteady'][name]
            self.fem_consts_steady[name].value = data['steady'][name]

class PIDTerm(Term):
    """Term that implements PID controller for a single variable. This is a base
    class for PIDTerm that would be otherwise quite tedious to implement.

    typical usage:
    - define PID constants via function 'set_pid(p:float, i:float, d:float)'
    - set control interval for time adaptation strategy of the simulation solver 
      via method 'set_ctrl_interval([low:float, high:float]). These values will 
      control the size of the time step. If the output controler changes between 
      timesteps more than 'high' the time step will be decreased such that the 
      change falls below 'high'. If the output controler changes between timesteps 
      less than 'low' the time step will be allowed to increase if other 
      adaptation conditions allow it as well.
    - set update function for reference value via method 'set_reference(f:function)'
      For example in a form 'lambda t: 10+sin(t/1000)'
    - set update function for probe value via method 'set_probe(f:function)'
      For example in a form 'lambda t: probes.get_value("probe_name")'
    - set update tolerance via method 'set_converge_tol(tol:float)'. This is 
      used to check if the PID controler has converged for current time step. 
      This is the way to ensure enought numerical consistency of the diferent parts
      of the simulation. This is necessary for the PID controler because it is 
      in essence inplicit source term affecting the solution.
    - set limits of the control variable (output of PID controler) via method
      'set_output_limits([low:float, high:float])'. This is to limit the ouutput
      of the PID controler to a certain range. For example if the output is power, 
      it is limited by the maximum power that the system can provide.
    """
    
    def define_terms(self):

        self.k_p = 1.0
        self.k_i = 0.0
        self.k_d = 0.0
        self.converge_tol = sb.np.inf
        self.lims = [-sb.np.inf, sb.np.inf]
        self.abs_ctrl_interval = (15, 30)

        # update diference first
        self.new_constant('diff', 0.0)
        self.set_update('diff', self.diff_update, prevent_override=True)

        # update all internal (prevent_override=True) stuff
        self.new_constant('p', 1.0)
        self.set_update('p', self.p_update, prevent_override=True)
        self.new_constant('d', 1.0)
        self.set_update('d', self.d_update, prevent_override=True)
        self.new_constant('i', 1.0)
        self.set_update('i', self.i_update, prevent_override=True)

        # at last update the output value
        self.new_constant('output', 0.0)
        self.set_update('output', self.out_update, prevent_override=True)

    def set_pid(self, p, i, d):
        """Sets PID constants.

        Args:
            p (float): Proportional constant
            i (float): Integral constant
            d (float): Derivative constant
        """
        self.k_p = p
        self.k_i = i
        self.k_d = d

    def set_ctrl_interval(self, interval):
        """Sets absolute interval for the time step adaptation strategy.

        Args:
            interval (list): List of two floats [low, high]
        """
        self.abs_ctrl_interval = interval

    def set_probe(self, f):
        """Sets update function for probe value.

        Args:
            f (function): Update function for probe value
        """
        self.probe_f = f

    def set_reference(self, f):
        """Sets update function for reference value.

        Args:
            f (function): Update function for reference value
        """
        #FIXME: PID TERM - This should be constant so there is steady equivalent automaticaly
        self.ref_f = f

    def converged(self):
        """Checks if the PID controler has converged for current time step.

        Returns:
            bool: True if the PID controler has converged, False otherwise
        """
        d_diff  = self.fem_consts['diff'].value[0] - self.diff_update(self.sim.t.value)
        return abs(d_diff) < self.converge_tol

    def adaptation(self):
        """Calculates the adaptation factor for the time step.

        Returns:
            float: Adaptation factor
        """
        adi = abs(self.fem_consts['i'].value[0] - self.fem_consts['i'].value[1])
        if adi > self.abs_ctrl_interval[1]:
            ref_adi = 0.2*self.abs_ctrl_interval[0] + 0.8*self.abs_ctrl_interval[1]
            return (ref_adi/adi)
        elif adi < self.abs_ctrl_interval[0]:
            return 1/0.95
        else:
            return 1.0
    
    def set_converge_tol(self, tol):
        """Sets the tolerance for the PID controler convergence.

        Args:
            tol (float): Tolerance for the PID controler convergence
        """
        self.converge_tol = tol

    def set_output_limits(self, lims):
        """Sets the limits for the PID controler output.

        Args:
            lims (list): List of two floats [low, high]
        """
        assert len(lims) == 2, "'lims' must be a list of len = 2"
        self.lims = lims

    def diff_update(self, t):
        """Calculates the difference between reference and probe values.

        Args:
            t (float): Time at which the difference is being calculated
        """
        ref = self.ref_f(t)
        rel_val = self.probe_f(t)
        return ref - rel_val
    
    def p_update(self, t):
        """Calculates the proportional term of the PID controler.

        Args:
            t (float): Time at which the proportional term is being calculated
        """
        diff = self.fem_consts['diff'].value[0]
        return self.k_p*diff
    
    def d_update(self, t):
        """Calculates the derivative term of the PID controler.

        Args:
            t (float): Time at which the derivative term is being calculated
        """
        #FIXME: this should also conside both time points similar to i_update
        diff = self.fem_consts['diff'].value[0]
        diff_n = self.fem_consts['diff'].value[1]
        return self.k_d*(diff-diff_n)/(self.sim.dt.value)
    
    def i_update(self, t):
        """Calculates the integral term of the PID controler.

        Args:
            t (float): Time at which the integral term is being calculated
        """
        diff = self.fem_consts['diff'].value[0]
        diff_n = self.fem_consts['diff'].value[1]
        prev_i = self.fem_consts['i'].value[1]
        _di = self.k_i*diff*self.sim.theta*(self.sim.dt.value)
        _di_n = self.k_i*diff_n*(1-self.sim.theta)*(self.sim.dt.value)
        test_i = prev_i + _di + _di_n
        p = self.fem_consts['p'].value[0]
        d = self.fem_consts['d'].value[0]

        # if limits are OK accept full i-term
        test_out_val = p + test_i + d
        if test_out_val >= self.lims[0] and test_out_val <= self.lims[1]:
            return test_i

        # if over the limits check if p+d itself is OK ad adjust the i-term only
        if p+d >= self.lims[0] and p+d <= self.lims[1]:

            # if i-term causes UNDER limit increase it
            if test_out_val < self.lims[0]:
                new_i = test_i + (self.lims[0] - p + test_i + d)

            # if i-term causes OVER limit decrease it
            elif test_out_val > self.lims[1]:
                new_i = test_i - (p + test_i + d - self.lims[1])
            return new_i
        
        # if p+d is over the limit use previous i-term value
        else:
            return prev_i
        
    def get_current_pid_values(self):
        """Returns current values of PID constants.

        Returns:
            tuple: Tuple of three floats (p, i, d)
        """
        p = self.fem_consts['p'].value[0]
        i = self.fem_consts['i'].value[0]
        d = self.fem_consts['d'].value[0]
        return p, i, d

    def out_update(self, t):
        """Calculates the output value of the PID controler.

        Args:
            t (float): Time at which the output value is being calculated
        """
        p, i, d = self.get_current_pid_values()
        out = p + i + d
        if out < self.lims[0]:
            out = self.lims[0]
        elif out > self.lims[1]:
            out = self.lims[1]
        return out
    
    def __call__(self, T, x, t=None, domain=None):
        """Return ufl form that represents the specific output per unit of volume"""
        i = self.sim.subdomain_map[domain]
        Vc = self.sim.V_subdomain[i]
        return self.get_constant('output', t)/Vc
    
