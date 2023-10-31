from . material_base import (Material, 
                             Lagrange_property, 
                             Polynomial_property, 
                             PropertyUnits)

class Steel04(Material):
    def __init__(self, domain, name="Steel04"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 600], [20, 20], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 600], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [450, 450], PropertyUnits.cp), 
                         name=name)

class Cartridge_heated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 600], [2.0, 2.0], PropertyUnits.k),
                         rho = Lagrange_property(domain, [20, 600], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [450, 450], PropertyUnits.cp), 
                         name=name)

class Cartridge_unheated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 600], [2.0, 2.0], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 600], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [450, 450], PropertyUnits.cp),
                         name=name)

class Standard_insulation(Material):
    def __init__(self, domain, name="Standard_insulation"):
        super().__init__(h0_T_ref = 20,  
                         k = Lagrange_property(domain, [20, 600], [0.04, 0.04], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 600], [40.0, 40.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [1200, 1200], PropertyUnits.cp), 
                         name=name)

class VIP(Material):
    def __init__(self, domain, name="VIP"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 600], [0.008, 0.008], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 600], [190, 190], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [800, 800], PropertyUnits.cp), 
                         name=name)

class Constant_sand(Material):
    def __init__(self, domain, name="Constant_sand"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 600], [0.3, 0.3], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 600], [1500.0, 1500], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [830, 830], PropertyUnits.cp), 
                         name=name)

class Sand(Material):
    def __init__(self, domain, name="Sand"):
        sigma = 5.670374419e-08
        eps = 0.9
        d = 0.0015
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 600], [0.42, 0.7], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 600], [2650.0, 2650.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [830, 830], PropertyUnits.cp), 
                         name=name)