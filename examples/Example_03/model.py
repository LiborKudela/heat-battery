from heat_battery.simulations.simulation_base import Simulation, fem, np, MPI
from heat_battery.simulations.probing import FunctionSampler
from heat_battery.simulations import terms
import plotly.graph_objects as go
import plotly.express as px
import os

class THW_twowire(Simulation):
    def define_form_subdomain_terms(self):

        wire_heating = terms.SerialWiresResistiveHeating(self)
        self.set_source_term(wire_heating, 'long_wire')
        self.set_source_term(wire_heating, 'short_wire')

        self.set_bc_term(terms.AmbientCooling(self), 'outer_surface')

    def create_common_probes(self, probes):

        self.T_probes_coords = list(self.geo_meta['points']['T'].values())
        self.T_probes_names = list(self.geo_meta['points']['T'].keys())
        sampler = FunctionSampler(self.T_probes_coords, self.domain)
        @probes.register_probe('T', '°C')
        def T_probe():
            return sampler.eval(self.T)
        
        h_form = 0
        for i, mat in enumerate(self.mats, 1):
            h_form += mat.h(self.T)*mat.rho(self.T)*self.jac*self.dx(i)
        H_form = fem.form(h_form)
        @probes.register_probe('heat', 'J')
        def evaluate():
            H = fem.assemble_scalar(H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H

        obj = self.get_source_term('long_wire')
        long_R = obj.dR_dx(self.T, 'long_wire')*self.jac*self.get_measure_dx('long_wire')
        form_long_R = fem.form(long_R)
        @probes.register_probe('long_R', 'mOhm')
        def evaluate():
            R = fem.assemble_scalar(form_long_R)
            R = self.domain.comm.allreduce((R), op=MPI.SUM)
            return R*1000
        
        obj = self.get_source_term('short_wire')
        short_R = obj.dR_dx(self.T, 'short_wire')*self.jac*self.get_measure_dx('short_wire')
        form_short_R = fem.form(short_R)
        @probes.register_probe('short_R', 'mOhm')
        def evaluate(): #type: ignore
            R = fem.assemble_scalar(form_short_R)
            R = self.domain.comm.allreduce((R), op=MPI.SUM)
            return R*1000
        
        @probes.register_probe('imag_R', 'mOhm')
        def evaluate():
            long_R, short_R = probes.get_values(['long_R','short_R'])
            return long_R - short_R
        
        @probes.register_probe('R', 'mOhm')
        def evaluate():
            long_R, short_R = probes.get_values(['long_R','short_R'])
            return long_R + short_R
        
        @probes.register_probe('I', 'A')
        def evaluate():
            return obj.fem_consts['current'].value[0]
        
        @probes.register_probe('long_U', 'V')
        def evaluate():
            current, long_R = probes.get_values(['I','long_R'])
            delta_U = current*long_R/1000
            return delta_U
        
        @probes.register_probe('short_U', 'V')
        def evaluate():
            current, short_R = probes.get_values(['I','short_R'])
            delta_U = current*short_R/1000
            return delta_U

        long_wire_A = np.pi*self.geo_meta['call_data']['custom_data']['long_wire']['diameter']**2/4
        long_wire_l = self.geo_meta['call_data']['custom_data']['long_wire']['length']
        @probes.register_probe('long_TfR', '°C')
        def evaluate():
            r = probes.get_value('long_R')/1000*long_wire_A/long_wire_l
            all_roots = self.mats['long_wire'].sigma.evaluate_roots(r)
            real_root = all_roots[np.isreal(all_roots)]
            return real_root[0].real
        
        short_wire_A = np.pi*self.geo_meta['call_data']['custom_data']['short_wire']['diameter']**2/4
        short_wire_l = self.geo_meta['call_data']['custom_data']['short_wire']['length']
        @probes.register_probe('short_TfR', '°C')
        def evaluate():
            r = probes.get_value('short_R')/1000*short_wire_A/short_wire_l
            all_roots = self.mats['short_wire'].sigma.evaluate_roots(r)
            real_root = all_roots[np.isreal(all_roots)]
            return real_root[0].real
        
        imag_wire_A = long_wire_A
        imag_wire_l = long_wire_l - short_wire_l
        @probes.register_probe('imag_TfR', '°C')
        def evaluate():
            r = probes.get_value('imag_R')/1000*imag_wire_A/imag_wire_l
            all_roots = self.mats['long_wire'].sigma.evaluate_roots(r)
            real_root = all_roots[np.isreal(all_roots)]
            return real_root[0].real
        
        long_power_form = fem.form(obj(self.T, self.x, self.t, 'long_wire')*self.jac*self.get_measure_dx('long_wire'))
        @probes.register_probe('long_power', 'W')
        def evaluate():
            q_power = fem.assemble_scalar(long_power_form)
            q_power = self.domain.comm.allreduce((q_power), op=MPI.SUM)
            return q_power
        
        short_power_form = fem.form(obj(self.T, self.x, self.t, 'short_wire')*self.jac*self.get_measure_dx('short_wire'))
        @probes.register_probe('short_power', 'W')
        def evaluate():
            q_power = fem.assemble_scalar(short_power_form)
            q_power = self.domain.comm.allreduce((q_power), op=MPI.SUM)
            return q_power
        
        @probes.register_probe('total_power', 'W')
        def evaluate():
            lp, sp = probes.get_values(['long_power','short_power'])
            return lp + sp
        
        total_wire_l = short_wire_l+long_wire_l
        @probes.register_probe('ql', 'W/m') 
        def evaluate():
            return probes.get_value('total_power')/total_wire_l
        
        @probes.register_probe('ql_mid', 'W/m')
        def evaluate():
            T_mid = probes.get_value('T')[0]
            I = probes.get_value('I')
            r = self.mats['long_wire'].sigma.evaluate(T_mid)
            ql = I**2*r/long_wire_A
            return ql

        self.R_std = 1.0
        @probes.register_probe('U_std', 'V')
        def evaluate():
            I_std = probes.get_value('I')
            return self.R_std*I_std
            
        @probes.register_probe('delta_Rw', 'mOhm')
        def evaluate():
            delta_Rw = probes.get_value('imag_R') - self.imag_R0
            return delta_Rw

        @probes.register_probe('U_b', 'mV')
        def evaluate():
            delta_Rw, I_std = probes.get_values(['delta_Rw', 'I'])
            return 0.5*I_std*delta_Rw
            
        bc_obj = self.get_bc_term('outer_surface')
        qloss_form = fem.form(bc_obj(self.T, self.x, self.t)*self.jac*self.get_measure_ds('outer_surface'))
        @probes.register_probe('heat loss', 'W')
        def evaluate():
            q_flow = fem.assemble_scalar(qloss_form)
            q_flow = self.domain.comm.allreduce((q_flow), op=MPI.SUM)
            return q_flow

    def calculate_lmbd(self, res):
        if self.domain.comm.rank == 0:
            res['t_sim_log'] = np.log(res['t_sim'])
            res['der_R'] = res[f'long_R'].diff()/res['t_sim_log'].diff()
            for i in range(len(self.T_probes_coords)):
                res[f'der_{i}'] = res[f'T[{i}]'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}'] = res[f'der_{i}'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}_rolling'] = res[f'2nd_der_{i}'].rolling(3).mean()
                res[f'abs_der_{i}'] = res[f'der_{i}'].abs()
                res[f'lmbd_{i}'] = res['ql']/(4*np.pi*res[f'der_{i}'])
            res[f'der_imag_TfR'] = res[f'imag_TfR'].diff()/res['t_sim_log'].diff()
            res[f'lmbd_imag_TfR'] = res['ql']/(4*np.pi*res[f'der_imag_TfR'])
            return res

    def run_experiment(self,
            T_amb_t=lambda t: 20, 
            alpha_t=lambda t: 10,
            P=0.12,
            R_std=1.0,
            **kwargs):
        
        # change update function in the wire
        def constQ_update(t):
            return np.sqrt(P/self.unsteady_probes.get_value('R')*1000)

        self.get_source_term('long_wire').set_update('current', constQ_update)
        self.get_bc_term('outer_surface').set_update('alpha', alpha_t)
        self.get_bc_term('outer_surface').set_update('T_amb', T_amb_t)
        self.unsteady_probes.evaluate_probes()
        self.imag_R0 = self.unsteady_probes.get_value('imag_R')
        self.R_std = R_std
        res = self.solve_unsteady(**kwargs)
        if self.domain.comm.rank == 0:
            result_dir = os.path.join(self.result_dir, f'{self.model_name}')
            res = self.calculate_lmbd(res.df)
            res.to_csv(os.path.join(result_dir, 'LMBD_result.csv'))
            
    def plot_data(self):
        if self.domain.comm.rank == 0:
            fig = go.Figure()
            if not hasattr(self.probes, 'df'):
                return fig
            res = self.probes.df.copy()
            res = self.calculate_lmbd(res)

            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['t_sim'], mode='lines', name='t_sim'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['dt'], mode='lines', name='dt'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['lmbd_imag_TfR'], mode='lines', name='lmbd_imag_TfR'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['long_R'], mode='lines', name='long_R'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['imag_TfR'], mode='lines', name='imag_TfR'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['der_R'], mode='lines', name='der_R'))
            fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['U_b'], mode='lines', name='U_b'))
            # fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['U_b_noisy'], mode='lines', name='U_b'))
            # fig.add_trace(go.Scatter(x=res['t_sim_log'], y=res['U_std'], mode='lines', name='U_std'))
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

