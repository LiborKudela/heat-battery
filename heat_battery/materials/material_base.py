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

    def plot(self, save_name=None, show=False, T_lim_used=None):
        if MPI.COMM_WORLD.rank == 0:
            if T_lim_used is None:
                T_lim_used = self.x_values[0], self.x_values[-1]
            T = np.arange(T_lim_used[0], T_lim_used[1], 1.0)
            k = np.polyval(np.flip(self.fem_const.value), T)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=T, y=k, mode='lines', name="poly"))
            #fig.add_trace(go.Scatter(x=self.x_values, y=self.y_values, name="L points"))

            fig.update_layout(
                xaxis_title='Temperature [°C]',
                yaxis_title=self.unit['name'] + ' ' + self.unit['unit'])

            if show:
                fig.show()
            if save_name is not None:
                fig.write_html(f'{save_name}.html')
                fig.write_image(f'{save_name}.jpg', scale=3)

class Polynomial_property(Material_property):
    def __init__(self, domain, c=[1,1,1], unit=PropertyUnits.default):
        self.domain = domain
        self.fem_const = fem.Constant(domain, PETSc.ScalarType(c))
        self.unit = unit
        self.order = len(c)-1

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
        return e

class Lagrange_property(Material_property):
    def __init__(self, domain, x=[0.0, 1.0], y=[1.0, 1.0], unit=PropertyUnits.default):
        self.domain = domain
        self.x_values = x
        self.y_values = y
        self.unit = unit

        self.order = len(x)-1
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
        return e

    def update_fem_const(self):
        coefs = np.flip(lagrange(self.x_values, self.y_values).coef)
        for i, coef in enumerate(coefs):
            self.fem_const.value[i] = coef

    def to_polynomial_property(self):
        return Polynomial_property(self.domain, c=self.fem_const.value.copy(), unit=self.unit)
    
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
        return e
    
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
                yaxis_title='Heat conductivity [W/(m.K)]')
            if show:
                fig.show()
            if save_name is not None:
                fig.write_html(f'{save_name}.html')
                fig.write_image(f'{save_name}.jpg', scale=2)
