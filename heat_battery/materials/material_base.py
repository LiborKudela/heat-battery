from mpi4py import MPI
from dolfinx import fem
from petsc4py import PETSc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import lagrange
from typing import List
import math

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
        fig = self.get_figure(T_lim=T_lim)
        fig.update_layout(
            xaxis_title='Temperature [°C]',
            yaxis_title=f"{self.unit['name']} {self.unit['unit']}",
            uirevision = "None",
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

        self.transform_jac = np.identity(self.n_values)

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

        self.fem_const = fem.Constant(domain, PETSc.ScalarType([0.0]*self.n_values))
        self.transform_jac = self.langrange_transform_matrix(self.x_values)
        self.update_fem_const()

    def langrange_transform_matrix(self, x_vec):
        n = len(x_vec)
        columns = [None]*n
        for i in range(n):
            columns[i] = x_vec**i
        return np.linalg.inv(np.column_stack(columns))

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
        self.fem_const.value[:] = self.transform_jac.dot(self.y_values)

    def to_polynomial_property(self):
        return Polynomial_property(self.domain, c=self.fem_const.value.copy(), unit=self.unit, multiplier=self.multiplier)
        
    def get_figure(self, T_lim=None):
        fig = go.Figure()
        T_lim = T_lim or (0, 1000)
        x = np.arange(T_lim[0], T_lim[1], 1.0)
        y = np.polyval(np.flip(self.fem_const.value), x)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name="poly"))
        fig.add_trace(go.Scatter(x=self.x_values, y=self.y_values, mode='markers', name="L - points"))
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

class MaterialsSet():
    def __init__(self, domain, mat_list):
        self.mat_dict = mat_list
        self.mats = self.construct_materials(domain, mat_list)

    def construct_materials(self, domain, mat_dict) -> List[Material]:
        return [constructor(domain, name=name) for constructor, name in mat_dict]

    def __getitem__(self, i):
        return self.mats[i]
    
    def __setitem__(self, i, value):
        self.mats[i] = value

    def __len__(self):
        return len(self.mats)
    
    def plot_property(self, m=None, property='k'):
        assert isinstance(m, int), "m must be an Integer"
        if MPI.COMM_WORLD.rank == 0:
            match property:
                case 'k':
                    fig = self.mats[m].k.plot(return_fig=True)
                case 'rho':
                    fig = self.mats[m].rho.plot(return_fig=True)
                case 'cp':
                    fig = self.mats[m].cp.plot(return_fig=True)
            return fig
            

    



                         

