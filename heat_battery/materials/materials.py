from . material_base import (Material, 
                             Lagrange_property, 
                             Polynomial_property, 
                             PropertyUnits,
                             MaterialsSet)

class Kannthal_A1(Material):
    def __init__(self, domain, name="Kanthal"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [50, 600, 800, 1000, 1200, 1400], [11, 20, 22, 26, 27, 35], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20], [1665], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20], [140], PropertyUnits.cp), 
                         name=name)

class TantalumWireConstant(Material):
    def __init__(self, domain, name="TantalumWire"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20], [59.4], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20], [1665], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20], [138], PropertyUnits.cp), 
                         name=name)
        
class TantalumWire(Material):
    def __init__(self, domain, name="TantalumWire"):
        #cp_poly_kelvin = [23.2424, 0.0073595, -1.086909e-06, -1.605216e-09, 5.7408e-13] #https://doi.org/10.1007/s11669-018-0627-2
        cp_poly_celsius = [25.1420336649887, 0.00645322061428832, -2.1453074208672e-6, -9.77976192e-10, 5.7408e-13] # calculated from cp_poly_kelvin
        #sigma_kelvin = [-1.55086538527354e-8, 5.27128094663826e-10, -8.09468841923354e-14, 1.07322020488636e-17] #https://doi.org/10.1063/1.555723 #corrected values
        sigma_celsius = [1.22655585494745e-7, 4.85309030315519e-10, -7.21523812233941e-14, 1.07322020488636e-17] # calculated from sigma_kelvin
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20], [59.4], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20], [1665], PropertyUnits.rho),
                         cp = Polynomial_property(domain, cp_poly_celsius, PropertyUnits.cp),#.to_lagrange_property([15, 750, 1500, 2300, 2900]), 
                         sigma = Polynomial_property(domain, sigma_celsius, PropertyUnits.sigma),
                         name=name)

class Steel04(Material):
    def __init__(self, domain, name="Steel04"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 1000], [16, 16], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 1000], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 1000], [450, 450], PropertyUnits.cp), 
                         name=name)
        
class Copper(Material):
    def __init__(self, domain, name="Copper"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [0, 20, 727], [401, 398, 357],  PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 1083], [893, 794], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20], [385], PropertyUnits.cp), 
                         name=name)

class Cartridge_heated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 1000], [2.0, 2.0], PropertyUnits.k),
                         rho = Lagrange_property(domain, [20, 1000], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 1000], [450, 450], PropertyUnits.cp), 
                         name=name)

class Cartridge_unheated(Material):
    def __init__(self, domain, name="Cartridge_heated"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20], [2.0], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 1000], [7850, 7850], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 1000], [450, 450], PropertyUnits.cp),
                         name=name)

class Standard_insulation(Material):
    def __init__(self, domain, name="Standard_insulation"):
        super().__init__(h0_T_ref = 20,  
                         k = Lagrange_property(domain, [20, 1000], [0.04, 0.04], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 1000], [40.0, 40.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 1000], [1200, 1200], PropertyUnits.cp), 
                         name=name)
        
class Standard_insulation_2(Material):
    def __init__(self, domain, name="Standard_insulation"):
        super().__init__(h0_T_ref = 20,  
                         k = Lagrange_property(domain, [20, 1000], [0.04, 0.06], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 1000], [40.0, 40.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 1000], [1200, 1200], PropertyUnits.cp), 
                         name=name)

class VIP(Material):
    def __init__(self, domain, name="VIP"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 1000], [0.008, 0.008], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 1000], [190, 190], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 1000], [800, 800], PropertyUnits.cp), 
                         name=name)

class Constant_sand(Material):
    def __init__(self, domain, name="Constant_sand"):
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20], [0.3], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20], [1500], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20], [830], PropertyUnits.cp), 
                         name=name)

class Sand(Material):
    def __init__(self, domain, name="Sand"):
        sigma = 5.670374419e-08
        eps = 0.9
        d = 0.0015
        super().__init__(h0_T_ref = 20, 
                         k = Lagrange_property(domain, [20, 300, 600], [0.42, 0.5, 0.6], PropertyUnits.k), 
                         rho = Lagrange_property(domain, [20, 600], [2650.0, 2650.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [830, 830], PropertyUnits.cp), 
                         name=name)
        
class SandTheory(Material):
    def __init__(self, domain, name="SandTheory"):
        sigma = 5.670374419e-08
        eps = 0.9
        d = 0.0015
        kc = [0.4+d*eps*4*sigma*273.15**3, d*eps*4*sigma*3*273.15**2, d*eps*4*sigma*3*273.15, d*eps*4*sigma]
        super().__init__(h0_T_ref = 20,
                         k = Polynomial_property(domain, kc, PropertyUnits.k).to_lagrange_property([20, 200, 400, 600]),
                         rho = Lagrange_property(domain, [20, 600], [2650.0, 2650.0], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20, 600], [830, 830], PropertyUnits.cp),
                         name=name)
        
class Contact_sand(Material):
    def __init__(self, domain, d=0.000001, name="Contact sand-cartridge"):
        super().__init__(h0_T_ref = 20,
                         k = Lagrange_property(domain, [20, 1000], [0.2, 0.2], PropertyUnits.K, multiplier=d*1000),
                         rho = Lagrange_property(domain, [20], [1], PropertyUnits.rho),
                         cp = Lagrange_property(domain, [20], [1], PropertyUnits.cp), 
                         name=name)