import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
#from scipy.special import ellipk, ellipe
from mpmath import ellipk, ellipe, ellippi


class disk(object):
    def __init__(self, radius, density):
        self.a = radius
        self.sigma = density
        self.mass = np.pi * self.a ** 2 * self.sigma
        
    def cmpPotential_rz(self, r_in, z_in):
        """
        Compute gravitational potential at point of coordinates (r,0,z) in polar coordinates. Due to circular symmetry, it is correct for any theta. Assume that disk is in the (x,y) plane and center on (0,0)
        see Lass & Blitzer 1983
        """
        r, z = np.meshgrid(r_in, z_in)
        k = np.sqrt(4. * self.a * r / (z**2 + (a + r) ** 2))
        n2 = 4. * self.a * r / (a + r) ** 2
        if r <= self.a:
            fac_inout = np.pi * np.abs(z)
        else:
            fac_inout = 0
        fac_elle = -np.sqrt(z**2 + (a + r) ** 2)
        fac_ellk = -(a**2 - r**2) / np.sqrt(z**2 + (a + r) ** 2)
        fac_ellpi = -(a - r) / (a + r) * z**2 / np.sqrt(z**2 + (a + r) ** 2)
        Vrz = 2. * self.sigma * constants.G * (fac_inout + fac_elle * ellipe(k) + fac_ellk * ellipk(k) + fac_ellpi * ellippi(n2, k))

        if np.size(r_in) == 1 and np.size(z_in) == 1:
            Vrz = Vrz[0,0]
        elif np.size(r_in) == 1:
            Vrz = Vrz[0,:]
        elif np.size(z_in) == 1:
            Vrz = Vrz[:,0]
        else:
            pass
        return Vrz

    def plt_Vrz(self, rmin, rmax, nr, zmin, zmax, nz):
        """Plot potential in (r,z) slice"""
        raise NotImplementedError()

    def plt_Vr(self, rmin, rmax, nr, z):
        """Plot potential as a function of r at a given z"""
        raise NotImplementedError()

    def plt_Vz(self, zmin, zmax, nz, r):
        """Plot potential as a function of z at a given r"""
        raise NotImplementedError()
        

        
