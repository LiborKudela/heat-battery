from .simulation_base import Simulation, fem, PETSc, np, MPI, FunctionSampler
import plotly.graph_objects as go
import plotly.express as px

class Expression_electricQc_unsteady:
    def __init__(self, sim: Simulation):
        self.sim = sim
        self.wire_A = np.pi*self.sim.geo_meta['call_data']['custom_data']['d_wire']**2/4   # compiled in cannot change
        self.wire_l = self.sim.geo_meta['call_data']['custom_data']['l_wire']
        self.current = fem.Constant(self.sim.domain, PETSc.ScalarType((0.5, 0.5))) # can change value w/o recompilation (v, v_n)
        self.update = self.control_Qavg_const
        self.Q = 0.1
        
    def dR_dx(self, T):
        #dudx = I/A*r(T(x))
        i = self.sim.subdomain_map['wire']
        return 1/self.wire_A**2*self.sim.mats[i].sigma(T)
    
    def q_vol(self, T, t):
        #Q = I*U
        if t is self.sim.t:
            I = self.current[0]
        elif t is self.sim.t_n:
            I = self.current[1]
        return I**2*self.dR_dx(T)
    
    def __call__(self, T, t, x):
        # T, t, x must be in the signature coz the form constructor will insert them
        # you do not need to use them if not needed
        return self.q_vol(T, t)

    def update(self, t):
        pass

    def next_step(self):
        self.current.value[0] = self.current.value[1]

    def control_Qavg_const(self, t):
        if t is self.sim.t:
            self.current.value[0] = np.sqrt(self.Q/self.sim.probes.get_value('R')*1000)
        elif t is self.sim.t_n:
            self.current.value[1] = np.sqrt(self.Q/self.sim.probes.get_value('R')*1000)

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

class Expression_electricQc_steady_state:
    def __init__(self, sim):
        self.sim = sim
        self.wire_A = np.pi*self.sim.geo_meta['call_data']['custom_data']['d_wire']**2/4   # compiled in cannot change
        self.current = fem.Constant(self.sim.domain, PETSc.ScalarType((0.1))) # can change value w/o recompilation
        
    def dU_dx(self, T):
        #dudx = I/A*r(T(x))
        I = self.current
        i = self.sim.subdomain_map['wire']
        return I/self.wire_A**2*self.sim.mats[i].sigma(T)
    
    def q_vol(self, T):
        #Q = I*U
        I = self.current 
        return I*self.dU_dx(T)
    
    def __call__(self, T, x):
        # T, t, x must be in the signature coz the form constructor will insert them
        # you do not need to use them if not needed
        return self.q_vol(T)

    def update(self):
        pass

class Expression_Tamb_steady_state:
    def __init__(self, sim):
        self.sim = sim
        self.T_amb = fem.Constant(self.sim.domain, PETSc.ScalarType((20.0)))
        self.alpha = fem.Constant(self.sim.domain, PETSc.ScalarType((10.7)))
    
    def __call__(self, T, x):
        # T, x must be in the signature coz the form constructor will insert them
        # you do not need to use them if not needed
        return self.alpha*(T - self.T_amb)

    def update(self):
        pass

class Experiment_v2(Simulation):
    def define_form_subdomain_terms(self):
        self.set_unsteady_source_term(Expression_electricQc_unsteady, 'wire')
        self.set_steady_state_source_term(Expression_electricQc_steady_state, 'wire')
        self.set_unsteady_bc_term(Expression_Tamb_unsteady, 'outer_surface')
        self.set_steady_state_bc_term(Expression_Tamb_steady_state, 'outer_surface')

    def set_unsteady_current(self, value):
        obj = self.get_unsteady_source_term('wire')
        obj.current.value[:] = value

    def create_probes(self, probes):
        super().create_probes(probes)

        self.T_probes_coords = list(self.geo_meta['probes']['T'].values())
        self.T_probes_names = list(self.geo_meta['probes']['T'].keys())
        sampler = FunctionSampler(self.T_probes_coords, self.domain)
        @probes.register_probe('T', '°C')
        def Tc_probe():
            return sampler.eval(self.T)
        
        h_form = 0
        for i, mat in enumerate(self.mats, 1):
            h_form += mat.h(self.T)*mat.rho(self.T)*self.jac*self.dx(i)
        H_form = fem.form(h_form)
        @probes.register_probe('heat', 'J')
        def H_probe():
            H = fem.assemble_scalar(H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H

        obj = self.get_unsteady_source_term('wire')
        i = self.subdomain_map['wire']

        form_R = fem.form(obj.dR_dx(self.T)*self.jac*self.get_measure_dx('wire'))
        @probes.register_probe('R', 'mOhm')
        def resistance():
            R = fem.assemble_scalar(form_R)
            R = self.domain.comm.allreduce((R), op=MPI.SUM)
            return R*1000
        
        @probes.register_probe('I', 'A')
        def current():
            return obj.current.value[0]
        
        @probes.register_probe('U', 'V')
        def voltage():
            delta_U = probes.get_value('I')*(probes.get_value('R')/1000)
            return delta_U

        wire_A = np.pi*self.geo_meta['call_data']['custom_data']['d_wire']**2/4
        wire_l = self.geo_meta['call_data']['custom_data']['l_wire']
        @probes.register_probe('TfR', '°C')
        def temperature_from_R():
            r = probes.get_value('R')/1000*wire_A/wire_l
            all_roots = self.mats[i].sigma.evaluate_roots(r)
            real_root = all_roots[np.isreal(all_roots)]
            return real_root[0].real
        
        @probes.register_probe('rfR', 'Ohm*m*1e6')
        def resistivity():
            return 1e6*probes.get_value('R')/1000/wire_l*wire_A
        
        power_form = fem.form(obj(self.T, self.t, self.x)*self.jac*self.get_measure_dx('wire'))
        @probes.register_probe('power', 'W')
        def power():
            q_power = fem.assemble_scalar(power_form)
            q_power = self.domain.comm.allreduce((q_power), op=MPI.SUM)
            return q_power
        
        @probes.register_probe('ql', 'W/m')
        def ql():
            ql = self.probes.get_value('power')/wire_l
            return ql
        
        i = self.subdomain_map['wire']
        @probes.register_probe('ql_mid', 'W/m')
        def ql_mid():
            T_mid = self.probes.get_value('T')[0]
            I = self.probes.get_value('I')
            r = self.mats[i].sigma.evaluate(T_mid)
            ql = I**2*r/wire_A
            return ql
            
        bc_obj = self.get_unsteady_bc_term('outer_surface')
        qloss_form = fem.form(bc_obj(self.T, self.t, self.x)*self.jac*self.get_measure_ds('outer_surface'))
        @probes.register_probe('heat loss', 'W')
        def loss_probe():
            q_flow = fem.assemble_scalar(qloss_form)
            q_flow = self.domain.comm.allreduce((q_flow), op=MPI.SUM)
            return q_flow

    def run_experiment(self, res_path="results/probes_Q_const.csv", 
                       T_amb_t=lambda t: 20, alpha_t=lambda t: 10, P=0.12, **kwargs):
        self.get_unsteady_source_term('wire').Q = P
        self.get_unsteady_bc_term('outer_surface').alpha_t = alpha_t
        self.get_unsteady_bc_term('outer_surface').T_amb_t = T_amb_t
        res = self.solve_unsteady(**kwargs)
        if self.domain.comm.rank == 0:
            l_wire = self.geo_meta['call_data']['custom_data']['l_wire']
            ql = self.probes.get_value('ql')
            res['t_sim_log'] = np.log(res['t_sim'])
            res['der_R'] = res[f'R'].diff()/res['t_sim_log'].diff()
            for i in range(len(self.T_probes_coords)):
                res[f'der_{i}'] = res[f'T[{i}]'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}'] = res[f'der_{i}'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}_rolling'] = res[f'2nd_der_{i}'].rolling(3).mean()
                res[f'abs_der_{i}'] = res[f'der_{i}'].abs()
                res[f'lmbd_{i}'] = (ql)/(4*np.pi*res[f'der_{i}'])
            res[f'der_TfR'] = res[f'TfR'].diff()/res['t_sim_log'].diff()
            res[f'lmbd_TfR'] = (ql)/(4*np.pi*res[f'der_TfR'])
            res.to_csv(res_path)

            res = res[(res['2nd_der_1_rolling'].abs() < 0.003) & (res['t_sim_log'] > 1)]
            k = np.zeros(len(self.T_probes_coords) + 1)
            for i in range(len(self.T_probes_coords)):
                der = np.polyfit(res['t_sim_log'].to_numpy(), res[f'T[{i}]'].to_numpy(), 1)[0]
                k[i] = (ql)/(4*np.pi*der)
            der = np.polyfit(res['t_sim_log'].to_numpy(), res[f'TfR'].to_numpy(), 1)[0]
            k[-1] = (ql)/(4*np.pi*der)
            return k
            
    def plot_data(self):
        if self.domain.comm.rank == 0:
            fig = go.Figure()
            if not hasattr(self.probes, 'df'):
                return fig
            res = self.probes.df.copy()
            l_wire = self.geo_meta['call_data']['custom_data']['l_wire']
            ql = self.probes.get_value('ql')
            res['t_sim_log'] = np.log(res['t_sim'])
            res['der_R'] = res[f'R'].diff()/res['t_sim_log'].diff()
            for i in range(len(self.T_probes_coords)):
                res[f'der_{i}'] = res[f'T[{i}]'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}'] = res[f'der_{i}'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}_rolling'] = res[f'2nd_der_{i}'].rolling(3).mean()
                res[f'abs_der_{i}'] = res[f'der_{i}'].abs()
                res[f'lmbd_{i}'] = (ql)/(4*np.pi*res[f'der_{i}'])
            res[f'der_TfR'] = res[f'TfR'].diff()/res['t_sim_log'].diff()
            res[f'lmbd_TfR'] = (ql)/(4*np.pi*res[f'der_TfR'])
            #res = res[(res['2nd_der_1_rolling'].abs() < 0.01) & (res['t_sim_log'] > 0)]

            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['t_sim'], mode='lines', name=f't_sim'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['dt'], mode='lines', name=f'dt'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'lmbd_TfR'], mode='lines', name=f'lmbd_TfR'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'R'], mode='lines', name=f'R'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'TfR'], mode='lines', name=f'TfR'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'der_R'], mode='lines', name=f'der_R'))
            for i in range(len(self.T_probes_coords)):
                fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'lmbd_{i}'], mode='lines', name=f'lmbd_{i}'))
                fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'der_{i}'], mode='lines', name=f'der_{i}'))
                fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'2nd_der_{i}'], mode='lines', name=f'2nd_der_{i}'))
                fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'2nd_der_{i}_rolling'], mode='lines', name=f'2nd_der_{i}_rolling'))
                fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res[f'T[{i}]'], mode='lines', name=f'T[{i}]'))
            fig.update_layout(uirevision="sellected_data")
            return fig
        
    def plot_all_results(self):
        if self.domain.comm.rank == 0:
            fig = go.Figure()
            if not hasattr(self.probes, 'df'):
                return fig
            res = self.probes.df.copy()
            res = res.set_index('t_sim')
            fig = px.line(res)
            fig.update_layout(uirevision="all_data")
            return fig

