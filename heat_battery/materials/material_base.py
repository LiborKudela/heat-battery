from mpi4py import MPI
from dolfinx import fem
from petsc4py import PETSc
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Union

class PropertyUnits:
    default = {'name':'', 'unit': '[]'}
    k = {'name':'Heat Conductivity', 'unit': '[W/m.K]'}
    rho = {'name':'Mass density', 'unit': '[kg/m3]'}
    cp = {'name':'Specific heat capacity', 'unit': '[J/kg.K]'}
    K = {'name':'Thermal contact conductance', 'unit': '[kW/m2K]'}
    R = {'name':'Thermal contact resistance', 'unit': '[m2K/kW]'}
    T = {'name':'Temperature', 'unit': '[C]'}
    T_amb = {'name':'Absolute temperature', 'unit': '[K]'}
    sigma = {'name':'Resistivity', 'unit':'Ωm'}

class Material_property:

    def set_values(self, y_values) -> None:
        pass

    def set_value(self, i, y) -> None:
        pass

    def get_values(self) -> np.ndarray:
        pass

    def get_value(self, i) -> float:
        pass

    def get_form_constant(self) -> fem.Constant:
        pass

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

    def transform_values_to_form_coeficients(self, y_values):
        return self.transform_jac.dot(y_values)
    
    def transform_form_coefficients_to_values(self, c):
        return self.transform_jac.dot(c)

    def set_values(self, c_values):
        self.fem_const.value[:] = self.transform_values_to_form_coeficients(np.array(c_values))

    def set_value(self, i, c):
        self.fem_const.value[i] = c

    def get_values(self):
        return self.fem_const.value.copy()

    def get_value(self, i):
        return self.fem_const.value[i]
    
    def evaluate(self, T):
        return np.polyval(np.flip(self.fem_const.value), T)*self.multiplier
    
    def evaluate_roots(self, y0):
        p = np.poly1d(np.flip(self.fem_const.value))
        return (p - y0/self.multiplier).roots

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
        self.transform_jac_inv = np.linalg.inv(self.transform_jac)
        self.update_fem_const()

    def transform_values_to_form_coeficients(self, y_values):
        return self.transform_jac.dot(y_values)
    
    def transform_form_coefficients_to_values(self, c):
        return self.transform_jac_inv.dot(c)

    def langrange_transform_matrix(self, x_vec):
        n = len(x_vec)
        columns = [None]*n
        for i in range(n):
            columns[i] = x_vec**i
        return np.linalg.inv(np.column_stack(columns))

    def set_values(self, y_values):
        self.y_values[:] = np.array(y_values)
        self.fem_const.value[:] = self.transform_values_to_form_coeficients(self.y_values)

    def set_value(self, i, y):
        self.y_values[i] = y
        self.fem_const.value[:] = self.transform_values_to_form_coeficients(self.y_values)

    def get_values(self):
        return self.y_values.copy()

    def get_value(self, i):
        return self.y_values[i]
    
    def evaluate(self, T):
        return np.polyval(np.flip(self.fem_const.value), T)*self.multiplier
    
    def evaluate_roots(self, y0):
        return np.roots(np.flip(self.fem_const.value), y0/self.multiplier)
    
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
                 sigma=None,
                 h0_T_ref = 20, 
                 name="Unspecified"):
        
        self.h0_T_ref = h0_T_ref
        self.k = k
        self.rho = rho
        self.cp = cp
        self.sigma = sigma
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

    def get_property(self, property: str) -> Union[Polynomial_property, Lagrange_property]:
        "This alows getting property by name (string)"
        return getattr(self, property)
    
    def get_property_values(self, property: str) -> np.ndarray:
        "This alows getting values of property by name (string) of the property"
        return self.get_property(self, property).get_values()
    
    def set_property_values(self, property: str, values) -> None:
        "This alows setting values of property by name (string) of the property"
        self.get_property(self, property).set_values(values)

class MaterialsSet():
    def __init__(self, domain, mat_dict, h0_T_ref=20):
        self.mat_dict = mat_dict
        self.mats = self.instantiate_materials(domain, self.mat_dict, h0_T_ref=h0_T_ref)
        self.key_map = {name:i for i, name in enumerate(mat_dict.keys())}
        self.h0_T_ref = h0_T_ref

    def instantiate_materials(self, domain, mat_dict, h0_T_ref=None) -> List[Material]:
        "Takes list of material classes and instantiates all of them"
        if h0_T_ref is None:
            h0_T_ref = 20
        return [tuple_data[0](domain, name=name, h0_T_ref=h0_T_ref) for name, tuple_data in mat_dict.items()]
    
    def resolve_index(self, m=None):
        if m is None:
            m = np.arange(len(self.mats))
        else:
            m = self.convert_to_integer_index(m)
        m = np.atleast_1d(m)
        return m

    def set_property_values(
        self,
        values: List[int] | np.ndarray,
        property: str,
        m: Optional[int | str | List[Union[int, str]] | np.ndarray] = None,
    ) -> None:
        "Sets values of property (string) of sellected material given by index m (int)"

        m=self.resolve_index(m)   
        start_idx = 0
        for i in m:
            p = self.get_material(i).get_property(property)
            end_idx = start_idx + p.n_values
            p.set_values(values[start_idx:end_idx])
            start_idx = end_idx

    def get_property_values(
        self,
        property: str,
        m: Optional[int | str | List[Union[int, str]] | np.ndarray] = None,
    ) -> np.ndarray:
        "Gets values of property (string) of sellected material given by index m (int or str)"

        m=self.resolve_index(m) 
        k = []
        for i in m:
            p = self.get_material(i).get_property(property)
            k.append(p.get_values())
        return np.concatenate(k)

    def single_convert_to_integer_index(self, index) -> int:
        if np.issubdtype(type(index), np.integer):
            return index
        elif isinstance(index, str):
            return self.key_map[index]

    def convert_to_integer_index(self, input):
        "Translates an index"
        if hasattr(input, '__iter__') and not isinstance(input, str):
            output = []
            for sigle_index in input:
                output.append(self.single_convert_to_integer_index(sigle_index))
            return output
        else:
            return self.single_convert_to_integer_index(input)

    def __getitem__(self, i) -> Material:
        i = self.convert_to_integer_index(i)
        return self.mats[i]

    def __setitem__(self, i, value):
        i = self.convert_to_integer_index(i)
        self.mats[i] = value

    def get_material(self, index) -> Material:
        i = self.convert_to_integer_index(index)
        return self.mats[i]

    def __len__(self) -> int:
        return len(self.mats)

    def plot_property(
        self, m: Optional[Union[int, str]] = None, property: str = "k"
    ) -> Optional[go.Figure]:
    
        m=self.resolve_index(m) 
        if MPI.COMM_WORLD.rank == 0:
            figs = []
            for i, single_m in enumerate(m):
                # only rank=0 return fig, other return None
                p = self.get_material(single_m).get_property(property)
                fig = p.plot(return_fig=True)
                fig.update_layout(title=dict(text=self.mats[single_m].name, x=0.5))
                figs.append(fig)
            return figs
        else:
            return [None]*len(m)
