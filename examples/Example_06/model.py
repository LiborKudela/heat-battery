from heat_battery.simulations import terms
from heat_battery.simulations.probing import FunctionSampler
from heat_battery.simulations.simulation_base import MPI, Simulation, fem, pd, ufl, Probe_writer
from heat_battery.simulations.terms_base import Term
import heat_battery.data.meteodata as meteodata
import numpy as np

class HallC3(Term):

    def define_terms(self):

        # yearly heat demand 352.6632 [GJ]
        
        a, b, c = 37, 18, 8 # building size
        V = a*b*c
        rho = 1.33
        cp = 1006.0
        self.C = V*rho*cp
        self.K_C3 = 1.105*171.6 + 0.310*691.0 + 1.765*288.8 + 1.1640*99.6
        self.T_room_ref = 18.0
        self.abs_ctrl_interval = (0.025, 0.05)
        self.converge_tol_T_room = 0.05
        self.converge_tol_Q_amb = 2
        self.alpha_m_lims = (1.0, 20.0)
        self.heating_on = lambda t: 1
        self.max_bivalent_power = 1000
        self.max_mem_power = 1000
        
        self.new_constant('T_amb', self.T_room_ref)
        self.set_update('T_amb', lambda t: 20.0)

        self.new_constant('alpha_s', 5.0)
        self.set_update('alpha_s', lambda t: 5.0)
        
        self.new_constant('K', self.K_C3)
        self.set_update('K', lambda t: self.K_C3, prevent_override=True)

        self.new_constant('Q_amb', 0.0)
        self.set_update('Q_amb', self.Q_amb_update, prevent_override=True)
        
        self.new_constant('Q_s', 0.0)

        self.new_constant('alpha_m', 20.0)
        self.set_update('alpha_m', self.alpha_m_update, prevent_override=True)

        self.new_constant('Q_m', 0.0)

        self.new_constant('Q', 0.0)
        self.set_update('Q', lambda t: self.get_current_value('Q_s') + self.get_current_value('Q_m'), prevent_override=True)
        self.new_integral('Q_cumulative', 0.0, 'Q')

        self.new_constant('Q_used', 0.0)
        self.set_update('Q_used', lambda t: self.fem_consts['Q'].value[0]*self.heating_on(t), prevent_override=True)
        self.new_integral('Q_cumulative_used', 0.0, 'Q_used')

        self.new_constant('Q_loss', 0.0)
        self.set_update('Q_loss', lambda t: self.fem_consts['Q'].value[0]*(not self.heating_on(t)), prevent_override=True)
        self.new_integral('Q_cumulative_loss', 0.0, 'Q_loss')

        self.new_constant('Q_bivalent', 0.0)
        self.set_update('Q_bivalent', self.Q_bivalent_update, prevent_override=True)
        self.new_integral('Q_bivalent_cumulative', 0.0, 'Q_bivalent')

        self.new_constant('T_room', self.T_room_ref)
        self.set_update('T_room', self.T_room_update, prevent_override=True)

        self.Q_s_update = self.update_from_form_integral('Outer_surface')
        self.set_update('Q_s', self.Q_s_update, prevent_override=True)

        self.Q_m_update = self.update_from_form_integral('Membrane_surface')
        self.set_update('Q_m', self.Q_m_update, prevent_override=True)

    # static value setters
    def set_alpha_m_lims(self, lims):
        self.alpha_m_lims = lims

    def set_heating_on(self, heating_on_func):
        self.heating_on = heating_on_func

    def set_max_bivalent_power(self, value):
        self.max_bivalent_power = value

    def set_max_mem_power(self, value):
        self.max_mem_power = value

    def set_T_room_ref(self, value):
        self.T_room_ref = value

    # dynamic update callbacks
    def alpha_m_update(self, t):
        T_room = self.fem_consts['T_room'].value[0]
        T_amb = self.fem_consts['T_amb'].value[0]
        T_m = self.sim.unsteady_probes.get_value('T_avg_m')  #FIXME: this shoud be passed through setter
        Q_s = self.fem_consts['Q_s'].value[0]
        m_area = self.sim.get_surface_area('Membrane_surface')
        K = self.fem_consts['K'].value[0]
        Q_equitherm = K*(self.T_room_ref - T_amb)
        alpha_m_current = np.clip((Q_equitherm-Q_s)/(m_area*(T_m-T_room)), *self.alpha_m_lims)
        return alpha_m_current if self.heating_on(t) else self.alpha_m_lims[0]
    
    def Q_amb_update(self, t):
        T_room = self.fem_consts['T_room'].value[0]
        T_amb = self.fem_consts['T_amb'].value[0]
        K = self.fem_consts['K'].value[0]
        return K*(T_amb - T_room)
    
    def Q_bivalent_update(self, t):
        T_amb = self.fem_consts['T_amb'].value[0]
        Q_in_total_storage = self.fem_consts['Q'].value[0]
        K = self.fem_consts['K'].value[0]
        Q_equitherm = K*(self.T_room_ref - T_amb)
        Q_bivalent = Q_equitherm - Q_in_total_storage
        return np.clip(Q_bivalent, 0.0, self.max_bivalent_power) if self.heating_on(t) else 0

    def T_room_update(self, t):
        dQ_amb = self.get_integral_step('Q_amb')
        dQ = self.get_integral_step('Q')
        dQ_bivalent = self.get_integral_step('Q_bivalent')
        T_room_n = self.fem_consts['T_room'].value[1]
        return T_room_n + (dQ_amb + dQ + dQ_bivalent)/self.C
    
    def converged(self):
        dT_room = self.fem_consts['T_room'].value[0] - self.T_room_update(self.sim.t.value)
        dT_q_amb = self.fem_consts['Q_amb'].value[0] - self.Q_amb_update(self.sim.t.value)
        return abs(dT_room) < self.converge_tol_T_room and abs(dT_q_amb) < self.converge_tol_Q_amb

    def adaptation(self):
        adi = abs(self.fem_consts['T_room'].value[0] - self.fem_consts['T_room'].value[1])
        if adi > self.abs_ctrl_interval[1]:
            ref_adi = 0.2*self.abs_ctrl_interval[0] + 0.8*self.abs_ctrl_interval[1]
            return (ref_adi/adi)
        elif adi < self.abs_ctrl_interval[0]:
            return 1/0.95
        else:
            return 1.0

    def __call__(self, T, x, t=None, domain=None):
        if domain == "Outer_surface":
            return self.get_constant('alpha_s', t)*(T - self.get_constant('T_room', t))
        elif domain == "Membrane_surface":
            return self.get_constant('alpha_m', t)*(T - self.get_constant('T_room', t))
        else:
            raise Exception(f"unknown domain {domain}")

class C3_passive(Simulation):
    def define_form_subdomain_terms(self):

        # add dynamic source term for malapa cartridge
        self.set_source_term(terms.TemperatureLimitedUniformHeatSource(self), "Cartridge")

        # add dynamic boundary term for ambient cooling on the surface
        hr_obj = HallC3(self)
        self.set_bc_term(hr_obj, "Outer_surface")
        self.set_bc_term(hr_obj, "Membrane_surface")

    def create_common_probes(self, probes: Probe_writer):

        @probes.register_probe('t_sim_days', 'day')
        def scale_time():
            return probes.get_value('t_sim')/3600/24
        
        # power toggle state
        @probes.register_probe('t_cpu_relative', 's/day')
        def toggle_state():
            return probes.get_value('t_cpu')/probes.get_value('t_sim_days')
        
        # power toggle state
        @probes.register_probe('t_remain_2', 'h')
        def toggle_state():
            t_cpu = probes.get_value('t_cpu')
            progress = probes.get_value('progress')
            return t_cpu/(progress)*(100-progress)/3600

        # temperature in points (cartgridges)
        self.T_probes_coords = list(self.geo_meta["points"]["T"].values())
        self.T_probes_names = list(self.geo_meta["points"]["T"].keys())
        sampler = FunctionSampler(self.T_probes_coords, self.domain)
        @probes.register_probe("T", "°C")
        def Tc_probe():
            return sampler.eval(self.T)

        # form for calculating heat inside storage
        h_form = 0.0
        for i, mat in enumerate(self.mats, 1):
            h_form += mat.h(self.T)*mat.rho(self.T)*self.jac*self.dx(i)
        H_form = fem.form(h_form)
        @probes.register_probe("H_storage", "J")
        def H_probe():
            H = fem.assemble_scalar(H_form)
            H = self.domain.comm.allreduce((H), op=MPI.SUM)
            return H#*2.77777778e-7
        
        # Heat flow from surface of the storage to the room
        @probes.register_probe('Q_s', 'W')
        def loss_probe():
            return self.get_bc_term('Outer_surface').get_current_value('Q_s')
        
        # Heat flow from mem pipes of the storage to the room
        @probes.register_probe('Q_m', 'W')
        def loss_probe():
            return self.get_bc_term('Outer_surface').get_current_value('Q_m')
        
        # Total Heat flow from the storage to the room
        @probes.register_probe('Q_s2r_total', 'W')
        def Q_total_from_storage_to_room():
            q_flow = probes.get_value('Q_s') + probes.get_value('Q_m')
            return q_flow

        # Heat flow from the room to the environment
        @probes.register_probe('Q_amb', 'W')
        def Q_amb():
            return self.get_bc_term('Outer_surface').get_current_value('Q_amb')
        
        # Heat flow from the room to the environment
        @probes.register_probe('Q_bivalent', 'W')
        def Q_bivalent():
            return self.get_bc_term('Outer_surface').get_current_value('Q_bivalent')
        
        # Power from pv panels to the cartridges (time integral)
        @probes.register_probe('Q_pv', 'W')
        def Q_in():
            return self.get_source_term('Cartridge').get_current_value('Q_in')
        
        # Power from pv panels to the cartridges (time integral)
        @probes.register_probe('Q_c', 'W')
        def Qc():
            return self.get_source_term('Cartridge').get_current_value('Qc')
        
        # Heat flow from the room to the environment
        @probes.register_probe('H_bivalent', 'J')
        def H_bivalent():
            return self.get_bc_term('Outer_surface').get_current_value('Q_bivalent_cumulative')
        
        # Total energy passed to the room (time integral)
        @probes.register_probe('H_s2r_total', 'J')
        def heat_from_storage_to_room():
            return self.get_bc_term('Outer_surface').get_current_value('Q_cumulative')
        
        # Total energy passed to the room when heating season active (time integral)
        @probes.register_probe('H_s2r_used', 'J')
        def heat_from_storage_to_room():
            return self.get_bc_term('Outer_surface').get_current_value('Q_cumulative_used')
        
        # Total energy demand of the building
        @probes.register_probe('H_demand', 'J')
        def heat_from_storage_to_room():
            return probes.get_value('H_bivalent') + probes.get_value('H_s2r_used')
        
        # Total energy lost from storage when not needed for heating
        @probes.register_probe('H_s2r_loss', 'J')
        def heat_from_storage_to_room():
            total = self.get_bc_term('Outer_surface').get_current_value('Q_cumulative')
            used = self.get_bc_term('Outer_surface').get_current_value('Q_cumulative_used')
            return total - used
        
        # Energy passed from pv panels to the cartridges
        @probes.register_probe('H_c', 'J')
        def Hc():
            return self.get_source_term('Cartridge').get_current_value('Qc_cumulative')
        
        # Energy passed from pv panels to the cartridges
        @probes.register_probe('H_pv', 'J')
        def H_pv():
            return self.get_source_term('Cartridge').get_current_value('Q_in_cumulative')
        
        # Mean temparature of the storage surface
        area_surf = self.get_surface_area('Outer_surface')
        Tmean_surf = fem.form(self.T/area_surf*self.jac*self.get_measure_ds('Outer_surface'))
        @probes.register_probe('T_avg_s', '°C')
        def average_temp():
            value = fem.assemble_scalar(Tmean_surf)
            value = self.domain.comm.allreduce((value), op=MPI.SUM)
            return value
        
        # Mean temperature of the mem pipes surfaces
        area_surf = self.get_surface_area('Membrane_surface')
        Tmean_m_surf = fem.form(self.T/area_surf*self.jac*self.get_measure_ds('Membrane_surface'))
        @probes.register_probe('T_avg_m', '°C')
        def average_temp():
            value = fem.assemble_scalar(Tmean_m_surf)
            value = self.domain.comm.allreduce((value), op=MPI.SUM)
            return value
        
        # Mean mixed tempearture of the room
        @probes.register_probe('T_avg_room', '°C')
        def average_temp():
            return self.get_bc_term('Outer_surface').get_current_value('T_room')
        
        # Mean temperature of the sand
        V_sand = self.get_subdomain_volume('Sand')
        T_mean_sand = fem.form(self.T/V_sand*self.jac*self.get_measure_dx('Sand'))
        @probes.register_probe('T_avg_sand', '°C')
        def sand_average_temp():
            value = fem.assemble_scalar(T_mean_sand)
            value = self.domain.comm.allreduce((value), op=MPI.SUM)
            return value

        # Ambient temperature at the given moment
        @probes.register_probe('T_amb', '°C')
        def average_temp():
            return self.get_bc_term('Outer_surface').get_current_value('T_amb')

        # Heating season
        @probes.register_probe('Heating_on', '-')
        def average_temp():
            return self.get_bc_term('Outer_surface').heating_on(self.t.value)
        
        # How much stored heat was not lost from storage (a.k.a. storage efficiency)
        @probes.register_probe('Eff_storage', '-')
        def average_temp():
            return (probes.get_value('H_c')-probes.get_value('H_s2r_loss'))/probes.get_value('H_c')
        
        # How much PV energy converted to heat was used for heating
        @probes.register_probe('Eff_used', '-')
        def average_temp():
            return probes.get_value('H_s2r_used')/probes.get_value('H_c')
        
        # How much energy generated by PV was used for heating
        @probes.register_probe('Eff_used_pv', '-')
        def average_temp():
            return probes.get_value('H_s2r_used')/probes.get_value('H_pv')

        # How much heat demand was coverd by the storage
        @probes.register_probe('Eff_demand_covered', '-')
        def average_temp():
            return probes.get_value('H_s2r_used')/probes.get_value('H_demand')
        
        # How much energy was generated with pv vs actual heat demand of building
        @probes.register_probe('Eff_pv_vs_demand', '-')
        def average_temp():
            return probes.get_value('H_pv')/probes.get_value('H_demand')
        
        # How much of generated PV power was converted to heat
        @probes.register_probe('Eff_pv_used', '-')
        def average_temp():
            return probes.get_value('H_c')/probes.get_value('H_pv')
        
        # PV system power toggle state
        @probes.register_probe('power_toggle', '-')
        def toggle_state():
            return self.get_source_term('Cartridge').get_current_value('toggle')

    def solve_steady(self, Qc=10, T_amb=20, xdmf_file=None, alpha=6.3):
        obj = self.get_source_term("Cartridge")
        obj.set_steady_state_value('Qc', Qc)

        obj = self.get_bc_term("Outer_surface")
        obj.set_steady_state_value('T_amb', T_amb)
        obj.set_steady_state_value('alpha', alpha)
        super().solve_steady(xdmf_file=xdmf_file)

        return pd.Series(
            data=self.steady_probes.get_value("T"),
            index=self.T_probes_names,
            name="Simulation",
        )

    def solve_unsteady(self,
            alpha_s=5.0,
            alpha_m_lims=(1.0, 30.0),
            location=meteodata.locations['Brno-FME'],
            pv_peak=1000,
            Tc_limit=500.0,
            max_bivalent_power=30000,
            max_mem_power=30000,
            **kwargs):
        
        meteo_loader = meteodata.CachedMeteoDataLoader('wheather')
        meteo_data = meteo_loader.fetch_hourly(location)[0]
        #self.next_event = lambda: (1+int(self.t_n.value/3600))-self.t_n.value
        
        offset_date = '2007-6-1'
        T_amb_t = meteodata.to_function(meteo_data, 'temp_air', offset_date=offset_date)
        P_pv_t = meteodata.to_function(meteo_data, 'P', offset_date=offset_date)
        P_pv_watts_t = lambda t: (pv_peak/1000)*P_pv_t(t)
        
        heating_season_t = meteodata.to_function(meteo_data, 'heating_season', offset_date=offset_date, interpolate=False)
        heating_pause_t = meteodata.to_function(meteo_data, 'heating_pause', offset_date=offset_date, interpolate=False)
        heating_on_t = lambda t: int(heating_season_t(t) and not heating_pause_t(t))
        
        # cartridge heating from PV panels
        obj_cartridge : terms.TemperatureLimitedUniformHeatSource
        obj_cartridge = self.get_source_term("Cartridge")
        obj_cartridge.set_update('T_probe', lambda t: self.unsteady_probes.get_value('T')[0])
        obj_cartridge.set_T_limit(Tc_limit)
        obj_cartridge.set_update('Q_in', P_pv_watts_t)

        # Building C3
        obj : HallC3
        obj = self.get_bc_term("Outer_surface") #heated room
        obj.set_initial_value('T_room', T_amb_t(0))
        obj.set_update('T_amb', T_amb_t)
        obj.set_update('alpha_s', lambda t: alpha_s)
        obj.set_alpha_m_lims(alpha_m_lims)
        obj.set_T_room_ref(18.0)
        obj.set_max_bivalent_power(max_bivalent_power)
        obj.set_max_mem_power(max_mem_power)
        obj.set_heating_on(heating_on_t)
  
        return super().solve_unsteady(
            **kwargs)
