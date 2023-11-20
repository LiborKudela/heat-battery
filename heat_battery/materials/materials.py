from . material_base import (Material, 
                             Lagrange_property, 
                             Polynomial_property, 
                             PropertyUnits,
                             MaterialsSet)

class Steel04(Material):
    def __init__(self, domain, name="Steel04"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20], [16], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [0, 1000], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 1000], [450, 450], PropertyUnits.cp), 
                         name=name)

class Cartridge_heated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20], [2.0], PropertyUnits.k),
                         rho = Lagrange_property(domain, [0, 1000], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 1000], [450, 450], PropertyUnits.cp), 
                         name=name)

class Cartridge_unheated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20], [2.0], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [0, 1000], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 1000], [450, 450], PropertyUnits.cp),
                         name=name)

class Standard_insulation(Material):
    def __init__(self, domain, name="Standard_insulation"):
        super().__init__(h0_T_ref = 20,  
                         k = Lagrange_property(domain, [0, 1000], [0.04, 0.06], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [0, 1000], [40.0, 40.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 1000], [1200, 1200], PropertyUnits.cp), 
                         name=name)

class VIP(Material):
    def __init__(self, domain, name="VIP"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [0, 1000], [0.008, 0.008], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [0, 1000], [190, 190], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 1000], [800, 800], PropertyUnits.cp), 
                         name=name)

class Constant_sand(Material):
    def __init__(self, domain, name="Constant_sand"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [0, 1000], [0.3, 0.3], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [0, 1000], [1500.0, 1500], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 1000], [830, 830], PropertyUnits.cp), 
                         name=name)

class Sand(Material):
    def __init__(self, domain, name="Sand"):
        sigma = 5.670374419e-08
        eps = 0.9
        d = 0.0015
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [0, 300, 600], [0.42, 0.5, 0.6], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [0, 600], [2650.0, 2650.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 600], [830, 830], PropertyUnits.cp), 
                         name=name)
        
class SandTheory(Material):
    def __init__(self, domain, name="SandTheory"):
        sigma = 5.670374419e-08
        eps = 0.9
        d = 0.0015
        kc = [0.4+d*eps*4*sigma*273.15**3, d*eps*4*sigma*3*273.15**2, d*eps*4*sigma*3*273.15, d*eps*4*sigma]
        super().__init__(h0_T_ref = 20,
                         k = Polynomial_property(domain, kc, PropertyUnits.k).to_lagrange_property([0, 200, 400, 600]),
                         rho = Lagrange_property(domain, [0, 600], [2650.0, 2650.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 600], [830, 830], PropertyUnits.cp),
                         name=name)
        
class Contact_sand(Material):
    def __init__(self, domain, d=0.0001, name="Contact sand-cartridge"):
        super().__init__(h0_T_ref = 20,
                         k = Lagrange_property(domain, [0, 1000], [0.1, 0.1], PropertyUnits.K, multiplier=d*1000),
                         rho = Lagrange_property(domain, [0, 1000], [40.0, 40.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [0, 1000], [1200, 1200], PropertyUnits.cp), 
                         name=name)