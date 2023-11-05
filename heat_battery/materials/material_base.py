from dolfinx import fem
from petsc4py import PETSc
import numpy as np
import plotly.graph_objects as go
from mpi4py import MPI
from scipy.interpolate import lagrange

class PropertyUnits:
    default = {'name':'', 'unit': '[]'}
    k = {'name':'Heat Conductivity', 'unit': '[W/m.K]'}
    rho = {'name':'Mass density', 'unit': '[kg/m3]'}
    cp = {'name':'Specific heat capacity', 'unit': '[J/kg.K]'}
    K = {'name':'Thermal contact conductance', 'unit': '[kW/m2K]'}
    R = {'name':'Thermal contact resistance', 'unit': '[m2K/kW]'}
    T = {'name':'Temperature', 'unit': '[C]'}
    T_amb = {'name':'Absolute temperature', 'unit': '[K]'}

class Material_property:

    def set_values(self, y_values) -> None:
        pass

    def set_value(self, i, y) -> None:
        pass

    def get_value(self, i) -> float:
        return 0.0

    def __call__(self, T):
        pass

    def get_figure(self, T_lim=None) -> go.Figure:
        pass

    def plot(self, T_lim=None, show=False, return_fig=False, save_name=None):
        if MPI.COMM_WORLD.rank == 0:
            fig = self.get_figure(T_lim=T_lim)
            fig.update_layout(
                xaxis_title='Temperature [°C]',
                yaxis_title=f"{self.unit['name']} {self.unit['unit']}",
                )

            if show:
                fig.show()
            if save_name is not None:
                fig.write_html(f'{save_name}.html')
                fig.write_image(f'{save_name}.jpg', scale=3)
            if return_fig:
                return fig

class Polynomial_property(Material_property):
    def __init__(self, domain, c=[1,1,1], unit=PropertyUnits.default, multiplier=1.0):
        self.domain = domain
        self.fem_const = fem.Constant(domain, PETSc.ScalarType(c))
        self.unit = unit
        self.n_values = len(c)
        self.order = len(c)-1
        self.multiplier = multiplier

    def set_values(self, c_values):
        self.fem_const.value[:] = np.array(c_values)

    def set_value(self, i, c):
        self.fem_const.value[i] = c

    def get_values(self):
        return self.fem_const.value.copy()

    def get_value(self, i):
        return self.fem_const.value[i]

    def __call__(self, T):
        e = 0
        for i in range(len(self.fem_const)):
            e += self.fem_const[i]*T**i
        return e*self.multiplier
    
    def to_lagrange_property(self, x):
        assert self.order <= len(x), f"x must be at least of len n_values - {self.n_values}"
        # TODO write test for this
        y = np.polyval(np.flip(self.fem_const.value), x)
        return Lagrange_property(self.domain, x=x, y=y, unit=self.unit)
    
    def get_figure(self, T_lim=None):
        if MPI.COMM_WORLD.rank == 0:
            fig = go.Figure()
            T_lim = T_lim or (0, 1000)
            x = np.arange(T_lim[0], T_lim[1], 1.0)
            y = np.polyval(np.flip(self.fem_const.value), x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name="poly"))
            return fig

class Lagrange_property(Material_property):
    def __init__(self, domain, x=[0.0, 1.0], y=[1.0, 1.0], unit=PropertyUnits.default, multiplier=1.0):
        assert len(x) == len(y), "x and y must be the same length"
        self.domain = domain
        self.x_values = np.array(x, dtype=float)
        self.y_values = np.array(y, dtype=float)
        self.unit = unit
        self.n_values = len(x)
        self.order = len(x)-1
        self.multiplier = multiplier

        self.fem_const = fem.Constant(domain, PETSc.ScalarType([0.0]*(self.order+1)))
        self.update_fem_const()

    def set_values(self, y_values):
        self.y_values[:] = np.array(y_values)
        self.update_fem_const()

    def set_value(self, i, y):
        self.y_values[i] = y
        self.update_fem_const()

    def get_values(self):
        return self.y_values.copy()

    def get_value(self, i):
        return self.y_values[i]
    
    def __call__(self, T):
        e = 0
        for i in range(len(self.fem_const)):
            e += self.fem_const[i]*T**i
        return e*self.multiplier

    def update_fem_const(self):
        coefs = np.flip(lagrange(self.x_values, self.y_values).coef)
        for i, coef in enumerate(coefs):
            self.fem_const.value[i] = coef

    def to_polynomial_property(self):
        return Polynomial_property(self.domain, c=self.fem_const.value.copy(), unit=self.unit)
        
    def get_figure(self, T_lim=None):
        if MPI.COMM_WORLD.rank == 0:
            fig = go.Figure()
            T_lim = T_lim or (0, 1000)
            x = np.arange(T_lim[0], T_lim[1], 1.0)
            y = np.polyval(np.flip(self.fem_const.value), x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name="poly"))
            fig.add_trace(go.Scatter(x=self.x_values, y=self.y_values, mode='markers', name="lagrange points"))
            return fig
    
class Material():
    def __init__(self,
                 k : Material_property, 
                 rho : Material_property, 
                 cp : Material_property,
                 h0_T_ref = 20, 
                 name="Unspecified"):
        
        self.h0_T_ref = h0_T_ref
        self.k = k
        self.rho = rho
        self.cp = cp
        self.name = name

    def h(self, T):
        # Enthalpy (cp integrated from h0_T_ref to T)
        e_offset = 0
        for i in range(len(self.cp.fem_const.value)):
            e_offset += self.h0_T_ref**(i+1)/(i+1)*self.cp.fem_const[i]
        e = -e_offset
        for i in range(len(self.cp.fem_const.value)):
            e += self.cp.fem_const[i]*T**(i+1)/(i+1)
        return e*self.cp.multiplier
    
    def plot_k(self, T_lim=(20, 600), T_lim_used=None, save_name=None, show=False):
        if MPI.COMM_WORLD.rank == 0:
            T = np.arange(T_lim[0], T_lim[1], 1.0)
            k = np.polyval(np.flip(self.k.fem_const.value), T)
            fig = go.Figure()
            fig.add_trace(go.Line(x=T, y=k))
            if T_lim_used is not None:
                fig.add_trace(go.Line(x=[T_lim_used[0], T_lim_used[0]], y=[np.min(k), np.max(k)], name="T_min"))
                fig.add_trace(go.Line(x=[T_lim_used[1], T_lim_used[1]], y=[np.min(k), np.max(k)], name="T_max"))
            fig.add_trace(go.Scatter(x=self.k.x_values, y=self.k.y_values, name="lagrange points"))
            fig.update_layout(
                title=self.name,
                xaxis_title='Temperature [°C]',
                yaxis_title=f"{self.k.unit['name']} {self.k.unit['unit']}")
            if show:
                fig.show()
            if save_name is not None:
                fig.write_html(f'{save_name}.html')
                fig.write_image(f'{save_name}.jpg', scale=2)

                         

