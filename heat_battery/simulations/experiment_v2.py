from .simulation_base import Simulation, fem, PETSc, np, MPI
from .probing import FunctionSampler
import plotly.graph_objects as go
import plotly.express as px
from . import unsteady_terms
from . import steady_terms

class Experiment_v2(Simulation):
    def define_form_subdomain_terms(self):

        # add dynamic source term for wire heating
        self.set_unsteady_source_term(unsteady_terms.SerialWiresResistiveHeating(self), 'wire')

        # add dynamic boundary term for ambient cooling on the surface
        self.set_unsteady_bc_term(unsteady_terms.AmbientCooling(self), 'outer_surface')
        
        # add static source term for wire heating
        self.set_steady_state_source_term(steady_terms.SerialWiresResistiveHeating(self), 'wire')

        # add dynamic boundary term for ambient cooling on the surface
        self.set_steady_state_bc_term(steady_terms.AmbientCooling(self), 'outer_surface')

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

        R = obj.dR_dx(self.T, 'wire')*self.jac*self.get_measure_dx('wire')
        form_R = fem.form(R)
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

        wire_A = np.pi*self.geo_meta['call_data']['custom_data']['wire']['diameter']**2/4
        wire_l = self.geo_meta['call_data']['custom_data']['wire']['length']
        @probes.register_probe('TfR', '°C')
        def temperature_from_R():
            r = probes.get_value('R')/1000*wire_A/wire_l
            all_roots = self.mats['wire'].sigma.evaluate_roots(r)
            real_root = all_roots[np.isreal(all_roots)]
            return real_root[0].real
        
        @probes.register_probe('rfR', 'Ohm*m*1e6')
        def resistivity():
            return 1e6*probes.get_value('R')/1000/wire_l*wire_A
        
        power_form = fem.form(obj(self.T, self.t, self.x, 'wire')*self.jac*self.get_measure_dx('wire'))
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
            r = self.mats['wire'].sigma.evaluate(T_mid)
            ql = I**2*r/wire_A
            return ql
            
        bc_obj = self.get_unsteady_bc_term('outer_surface')
        qloss_form = fem.form(bc_obj(self.T, self.t, self.x)*self.jac*self.get_measure_ds('outer_surface'))
        @probes.register_probe('heat loss', 'W')
        def loss_probe():
            q_flow = fem.assemble_scalar(qloss_form)
            q_flow = self.domain.comm.allreduce((q_flow), op=MPI.SUM)
            return q_flow

    def calculate_lmbd(self, res):
        if self.domain.comm.rank == 0:
            res['t_sim_log'] = np.log(res['t_sim'])
            res['der_R'] = res[f'R'].diff()/res['t_sim_log'].diff()
            for i in range(len(self.T_probes_coords)):
                res[f'der_{i}'] = res[f'T[{i}]'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}'] = res[f'der_{i}'].diff()/res['t_sim_log'].diff()
                res[f'2nd_der_{i}_rolling'] = res[f'2nd_der_{i}'].rolling(3).mean()
                res[f'abs_der_{i}'] = res[f'der_{i}'].abs()
                res[f'lmbd_{i}'] = res['ql']/(4*np.pi*res[f'der_{i}'])
            res[f'der_TfR'] = res[f'TfR'].diff()/res['t_sim_log'].diff()
            res[f'lmbd_TfR'] = res['ql']/(4*np.pi*res[f'der_TfR'])
            return res

    def run_experiment(self, res_path="results/probes_Q_const.csv", 
                       T_amb_t=lambda t: 20, alpha_t=lambda t: 10, P=0.12, **kwargs):
        self.get_unsteady_source_term('wire').Q = P
        self.get_unsteady_bc_term('outer_surface').alpha_t = alpha_t
        self.get_unsteady_bc_term('outer_surface').T_amb_t = T_amb_t
        res = self.solve_unsteady(**kwargs)
        if self.domain.comm.rank == 0:
            res = self.calculate_lmbd(res)
            res.to_csv(res_path)
            
    def plot_data(self):
        if self.domain.comm.rank == 0:
            fig = go.Figure()
            if not hasattr(self.probes, 'df'):
                return fig
            res = self.probes.df.copy()
            res = self.calculate_lmbd(res)

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

