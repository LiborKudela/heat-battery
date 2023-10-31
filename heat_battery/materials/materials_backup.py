from dolfinx import fem
from petsc4py import PETSc
import numpy as np
import plotly.graph_objects as go
from mpi4py import MPI

class Mat_property():
    def __init__(self, domain, c = [1,1,1]):
        self.fem_const = fem.Constant(domain, PETSc.ScalarType(c))
        self.value = self.fem_const.value #ref

    def __call__(self, T):
        # returns an ulf expresion
        e = 0
        for i in range(len(self.fem_const)):
            e += self.fem_const[i]*T**i
        return e
    
class Material():
    def __init__(self, domain=None, h0_T_ref = 20, k_c = [1,1,1], rho_c = [1,1,1], cp_c = [1,1,1], name="Unspecified"):
        self.h0_T_ref = h0_T_ref
        self.k = Mat_property(domain, c=k_c)
        self.rho = Mat_property(domain, c=rho_c)
        self.cp = Mat_property(domain, c=cp_c)
        self.name = name

    def h(self, T):
        # cp integrated from h0_T_ref to T
        e_offset = 0
        for i in range(len(self.cp.fem_const.value)):
            e_offset += self.h0_T_ref**(i+1)/(i+1)*self.cp.fem_const[i]
        e = -e_offset
        for i in range(len(self.cp.fem_const.value)):
            e += self.cp.fem_const[i]*T**(i+1)/(i+1)
        return e
    
    def plot(self, T_lim=(20, 600), T_lim_used=None, save_name=None, show=False):
        if MPI.COMM_WORLD.rank == 0:
            T = np.arange(T_lim[0], T_lim[1], 1.0)
            k = np.polyval(np.flip(self.k.value), T)
            fig = go.Figure()
            fig.add_trace(go.Line(x=T, y=k))
            if T_lim_used is not None:
                fig.add_trace(go.Line(x=[T_lim_used[0], T_lim_used[0]], y=[np.min(k), np.max(k)], name="T_min"))
                fig.add_trace(go.Line(x=[T_lim_used[1], T_lim_used[1]], y=[np.min(k), np.max(k)], name="T_max"))
            fig.update_layout(
                title=self.name,
                xaxis_title='Temperature [°C]',
                yaxis_title='Heat conductivity [W/(m.K)]')
            if show:
                fig.show()
            if save_name is not None:
                fig.write_html(f'{save_name}.html')
                fig.write_image(f'{save_name}.jpg', scale=2)
    
class Steel04(Material):
    def __init__(self, domain, name="Steel04"):
        super().__init__(domain=domain, h0_T_ref = 20, k_c = [45.51, -0.006], rho_c = [7850], cp_c = [450], name=name)

class Cartridge_heated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(domain=domain, h0_T_ref = 20, k_c = [2.0, 0.0], rho_c = [7850], cp_c = [450], name=name)

class Cartridge_unheated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(domain=domain, h0_T_ref = 20, k_c = [2.0, 0.0], rho_c = [7850], cp_c = [450], name=name)

class Standard_insulation(Material):
    def __init__(self, domain, name="Standard_insulation"):
        super().__init__(domain=domain, h0_T_ref = 20, k_c = [0.048, 0.0], rho_c = [40.0], cp_c = [1200], name=name)

class VIP(Material):
    def __init__(self, domain, name="VIP"):
        super().__init__(domain=domain, h0_T_ref = 20, k_c = [0.008], rho_c = [190], cp_c = [800], name=name)

class Constant_sand(Material):
    def __init__(self, domain, name="Constant_sand"):
        super().__init__(domain=domain, h0_T_ref = 20, k_c = [0.3], rho_c = [1500.0], cp_c = [830], name=name)

class Sand(Material):
    def __init__(self, domain, name="Sand"):
        sigma = 5.670374419e-08
        eps = 0.9
        d = 0.0015
        super().__init__(domain=domain, 
                         h0_T_ref = 20, 
                         k_c = [0.4+d*eps*4*sigma*273.15**3, d*eps*4*sigma*3*273.15**2, d*eps*4*sigma*3*273.15, d*eps*4*sigma], 
                         rho_c = [2650.0], 
                         cp_c = [830],
                         name=name)
