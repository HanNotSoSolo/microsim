"""
l = length of source cylinder (centered on z = 0)
L = length of target cylinder
zs = z-coordinate of target cylinder's center
kappa = sqrt(k^2 + 1/lambda^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1#, struve
from mpmath import struveh as struve
from scipy.integrate import quad, dblquad, trapezoid
from scipy.constants import G
from time import time


def getCase(l,L,zs):
    if l <= 0 or L <= 0:
        raise ValueError("l and L must be > 0")
    
    if zs - L >= -l and zs + L <= l:
        case = 'A'
    elif zs - L > l:
        case = 'B1'
    elif zs + L < -l:
        case = 'B2'
    elif zs - L > -l and zs + L > l:
        case = 'C1'
    elif zs - L < -l and zs + L < l:
        case = 'C2'
    elif zs - L < -l and zs + L > l:
        case = 'D'
    else:
        raise ValueError('Case not recognized!')
    return case


def H1(l,L,zs, kappa):
    if getCase(l,L,zs) in ['A', 'B2', 'C2']:
        val = 0
    elif getCase(l,L,zs) == 'B1':
        val = 4 * np.exp(-kappa * zs) / kappa * np.sinh(kappa * l) * np.sinh(kappa * L)
    elif getCase(l,L,zs) in ['C1', 'D']:
        val = -2./kappa * (np.exp(-kappa * (zs + L)) - np.exp(-kappa * l)) * np.sinh(kappa * l)
    else:
        raise ValueError('Case not recognized!')
    return val

def H2(l,L,zs, kappa):
    if getCase(l,L,zs) in ['B1', 'B2', 'D']:
        val = 0
    elif getCase(l,L,zs) == 'A':
        val = 4 * np.exp(-kappa * l) / kappa * np.sinh(kappa * L) * np.sinh(kappa * zs)
    elif getCase(l,L,zs) == 'C1':
        val = 2 * np.exp(-kappa * l) / kappa * (np.cosh(kappa * l) - np.cosh(kappa * (zs - L)))
    elif getCase(l,L,zs) == 'C2':
        val = 2 * np.exp(-kappa * l) / kappa * (np.cosh(kappa * (zs + L)) - np.cosh(kappa * l))
    else:
        raise ValueError('Case not recognized!')
    return val
    
def H3(l,L,zs, kappa):
    if getCase(l,L,zs) in ['A', 'B1', 'C1']:
        val = 0
    elif getCase(l,L,zs) == 'B2':
        val = -4 * np.exp(kappa * zs) / kappa * np.sinh(kappa * l) * np.sinh(kappa * L)
    elif getCase(l,L,zs) in ['C2', 'D']:
        val = -2./kappa * (np.exp(-kappa * l) - np.exp(kappa * (zs - L))) * np.sinh(kappa * l)
    else:
        raise ValueError('Case not recognized!')
    return val

def H4(l,L,zs, kappa):
    if getCase(l,L,zs) in ['B1', 'B2']:
        val = 0
    elif getCase(l,L,zs) == 'A':
        val = 4* L - 4 * np.exp(-kappa * l) / kappa * np.sinh(kappa * L) * np.cosh(kappa * zs)
    elif getCase(l,L,zs) == 'C1':
        val = 2 * (l + L - zs) - 2 * np.exp(-kappa * l) / kappa * (np.sinh(kappa * l) - np.sinh(kappa * (zs - L)))
    elif getCase(l,L,zs) == 'C2':
        val = 2 * (l + L + zs) - 2 * np.exp(-kappa * l) / kappa * (np.sinh(kappa * (zs + L)) + np.sinh(kappa * l))
    elif getCase(l,L,zs) == 'D':
        val = 4 * l - 4 * np.exp(-kappa * l) / kappa * np.sinh(kappa * l)
    else:
        raise ValueError('Case not recognized!')
    return val


def Rplus(theta, a, xs):
    if a <= 0:
        raise ValueError("a must be > 0")
    return xs * np.cos(theta) + np.sqrt(a**2 - xs**2 * np.sin(theta)**2)

def Rminus(theta, a, xs):
    if a <= 0:
        raise ValueError("a must be > 0")
    if np.abs(xs) > a:
        return xs * np.cos(theta) - np.sqrt(a**2 - xs**2 * np.sin(theta)**2)
    else:
        return 0


def Kz(k, a, xs, epsabs=1.49e-13, epsrel=1.49e-13, pltIntgd = False):
    #the R-integral is analytic
    Rint = lambda theta: (Rplus(theta, a, xs) * j1(k * Rplus(theta, a, xs)) - Rminus(theta, a, xs) * j1(k * Rminus(theta, a, xs))) / k

    #but the integral on theta is not easy, so do it numerically (unless xs = 0)
    if xs == 0:
        return 2 * np.pi * a * j1(k * a) / k
    
    if np.abs(xs) > a:
        theta_min = -np.arcsin(a / np.abs(xs)) * 0.999999 #0.99999 to avoid 0 in sqrt in integrand at the boundaries of integration
        theta_max = np.arcsin(a / np.abs(xs)) * 0.999999
        if xs < 0:
            theta_min += np.pi
            theta_max += np.pi
    else:
        theta_min = 0
        theta_max = 2. * np.pi

    if pltIntgd:
        th = np.linspace(theta_min, theta_max, 500)
        plt.plot(th, Rint(th))
        plt.xlabel('theta')
        plt.ylabel('Kz intgd')
        plt.show()
        
    res = quad(Rint, theta_min, theta_max, epsabs=epsabs, epsrel=epsrel)
    _checkIntegral(res, eps = 1e-2, origin = "Kz")
    #print(res)
    return res[0]

def Kz2D(k, a, xs, epsabs=1.49e-13, epsrel=1.49e-13, pltIntgd = False):
    """
    Cross-test. Kz with numeric integration on r. Must give same result as Kz.
    """
    intgd = lambda r, th: r * j0(k * r)
    gfun = lambda theta: Rminus(theta, a, xs)
    hfun = lambda theta: Rplus(theta, a, xs)
    
    if np.abs(xs) > a:
        theta_min = -np.arcsin(a / np.abs(xs)) * 0.999999 #0.99999 to avoid 0 in sqrt in integrand at the boundaries of integration
        theta_max = np.arcsin(a / np.abs(xs)) * 0.999999
        if xs < 0:
            theta_min += np.pi
            theta_max += np.pi
    else:
        theta_min = 0
        theta_max = 2. * np.pi

    if pltIntgd:
        th_span = theta_max - theta_min
        for theta in theta_min + np.array([0.1, 0.25, 0.5, 0.75, 0.9]) * th_span:
            R = np.linspace(Rminus(theta, a, xs), Rplus(theta, a, xs), 500)
            Rintgd = intgd(R, theta)
            plt.plot(R, Rintgd, label = 'th=' + str(theta))
        plt.legend()
        plt.xlabel('r')
        plt.ylabel('Kr integrand')
        plt.show()
        
    res = dblquad(intgd, theta_min, theta_max, gfun, hfun, epsabs=epsabs, epsrel=epsrel)
    _checkIntegral(res, eps = 1e-2, origin = "Kz2D")
    #print(res)
    return res[0]

def Kr(k, a, xs, epsabs=1.49e-13, epsrel=1.49e-13, pltIntgd = False):
    #the R-integral is analytic
    Rint = lambda theta: np.cos(theta) * np.pi / 2 * ( (Rplus(theta, a, xs) * j1(k * Rplus(theta, a, xs)) * struve(0, k * Rplus(theta, a, xs)) - Rplus(theta, a, xs) * j0(k * Rplus(theta, a, xs)) * struve(1, k * Rplus(theta, a, xs))) - (Rminus(theta, a, xs) * j1(k * Rminus(theta, a, xs)) * struve(0, k * Rminus(theta, a, xs)) - Rminus(theta, a, xs) * j0(k * Rminus(theta, a, xs)) * struve(1, k * Rminus(theta, a, xs))))
    #Rint = lambda theta: np.sign(np.cos(theta)) * np.pi / 2 * ( (Rplus(theta, a, xs) * j1(k * Rplus(theta, a, xs)) * struve(0, k * Rplus(theta, a, xs)) - Rplus(theta, a, xs) * j0(k * Rplus(theta, a, xs)) * struve(1, k * Rplus(theta, a, xs))) - (Rminus(theta, a, xs) * j1(k * Rminus(theta, a, xs)) * struve(0, k * Rminus(theta, a, xs)) - Rminus(theta, a, xs) * j0(k * Rminus(theta, a, xs)) * struve(1, k * Rminus(theta, a, xs))))

    if a == 0:
        return 0.
    
    #but the integral on theta is not easy, so do it numerically (unless xs = 0)
    if xs == 0:
        return 0.
    
    if np.abs(xs) > a:
        theta_min = -np.arcsin(a / np.abs(xs)) * 0.999999 #0.99999 to avoid 0 in sqrt in integrand at the boundaries of integration
        theta_max = np.arcsin(a / np.abs(xs)) * 0.999999
        if xs < 0:
            theta_min += np.pi
            theta_max += np.pi
    else:
        theta_min = 0
        theta_max = 2. * np.pi

    #the integrand is even wrt middle of range, so we can update theta_min
    theta_min = 0.5 * (theta_min + theta_max)
        
    if pltIntgd:
        th = np.linspace(theta_min, theta_max, 500)
        plt.plot(th, Rint(th))
        plt.xlabel('theta')
        plt.ylabel('Kr intgd, k = ' + str(k))
        plt.show()
        
    res = quad(Rint, theta_min, theta_max, epsabs=epsabs, epsrel=epsrel)
    _checkIntegral(res, eps = 1e-2, origin = "Kr")
    #print(res)
    return 2 * res[0] #*2 b/c even integrand

def Kr2D(k, a, xs, epsabs=1.49e-13, epsrel=1.49e-13, pltIntgd = False):
    intgd = lambda r, th: k * r * j1(k * r) * np.sign(np.cos(th))
    gfun = lambda theta: Rminus(theta, a, xs)
    hfun = lambda theta: Rplus(theta, a, xs)
    
    if np.abs(xs) > a:
        theta_min = -np.arcsin(a / np.abs(xs)) * 0.999999 #0.99999 to avoid 0 in sqrt in integrand at the boundaries of integration
        theta_max = np.arcsin(a / np.abs(xs)) * 0.999999
        if xs < 0:
            theta_min += np.pi
            theta_max += np.pi
    else:
        theta_min = 0
        theta_max = 2. * np.pi

    if pltIntgd:
        th_span = theta_max - theta_min
        for theta in theta_min + np.array([0.1, 0.25, 0.5, 0.75, 0.9]) * th_span:
            R = np.linspace(Rminus(theta, a, xs), Rplus(theta, a, xs), 500)
            Rintgd = intgd(R, theta)
            plt.plot(R, Rintgd, label = 'th=' + str(theta))
        plt.legend()
        plt.xlabel('r')
        plt.ylabel('Kr integrand')
        plt.show()
        
    res = dblquad(intgd, theta_min, theta_max, gfun, hfun, epsabs=epsabs, epsrel=epsrel)
    _checkIntegral(res, eps = 1e-2, origin = "Kr2D")
    #print(res)
    return res[0]

def Ksurf(a, xs):
    """
    Just a test of Kr (which also generalizes and Kz) with intgd = 1, so that the output must be pi * a^2 (the surface of the disk)
    """
    intgd = lambda r, th: r
    gfun = lambda theta: Rminus(theta, a, xs)
    hfun = lambda theta: Rplus(theta, a, xs)
    
    if np.abs(xs) > a:
        theta_min = -np.arcsin(a / np.abs(xs))
        theta_max = np.arcsin(a / np.abs(xs))
        if xs < 0:
            theta_min += np.pi
            theta_max += np.pi
    else:
        theta_min = 0
        theta_max = 2. * np.pi

    res = dblquad(intgd, theta_min, theta_max, gfun, hfun)
    return res


def k_intgd_z(k, xs, zs, ap, L, a, b, l, lmbda = 1):
    """
    xs, zs -- x and z coordinates of target cylinder's center (ys=0)
    ap -- target cylinder's radius
    L -- target cylinder's half-length
    a -- source cylinder's inner radius
    b -- source cylinder's outer radius
    l -- source cylinder's half-length
    lmbda -- Yukawa range
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    xyIntegral = Kz(k, ap, xs)
    zIntegral = H1(l,L,zs, kappa) + H2(l,L,zs, kappa) + H3(l,L,zs, kappa)
    sourceBessels = b * j1(k * b) - a * j1(k * a)
    intgd = (xyIntegral * sourceBessels * zIntegral) / kappa
    return intgd

def k_intgd_r(k, xs, zs, ap, L, a, b, l, lmbda = 1, pltKrIntgd = False):
    """
    xs, zs -- x and z coordinates of target cylinder's center (ys=0)
    ap -- target cylinder's radius
    L -- target cylinder's half-length
    a -- source cylinder's inner radius
    b -- source cylinder's outer radius
    l -- source cylinder's half-length
    lmbda -- Yukawa range
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    xyIntegral = Kr(k, ap, xs, pltIntgd = pltKrIntgd)
    zIntegral = H1(l,L,zs, kappa) + H4(l,L,zs, kappa) - H3(l,L,zs, kappa)
    sourceBessels = b * j1(k * b) - a * j1(k * a)
    intgd = (xyIntegral * sourceBessels * zIntegral) / kappa**2
    return intgd


def cmpFz(xs, zs, ap, L, rhop, a, b, l, rho, alpha = 1, lmbda = 1, kmax = 50, epsabs=1.49e-13, epsrel=1.49e-13, plotIntgd = False, nk4plot = 500):
    intgd = lambda k: k_intgd_z(k, xs, zs, ap, L, a, b, l, lmbda = lmbda)
    res = quad(intgd, 0, kmax, epsabs=epsabs, epsrel=epsrel)
    _checkIntegral(res, eps = 1e-2, origin = "cmpFz")
    k_int = res[0]
    fz = -2 * np.pi * G * rho * rhop * alpha * k_int
    #fz = -2 * G * rho * rhop * alpha * k_int
    #print('Fz...', fz)
    return fz

def cmpFr(xs, zs, ap, L, rhop, a, b, l, rho, alpha = 1, lmbda = 1, kmax = 50, epsabs=1.49e-13, epsrel=1.49e-13, plotIntgd = False, nk4plot = 500, pltKrIntgd = False):
    intgd = lambda k: k_intgd_r(k, xs, zs, ap, L, a, b, l, lmbda = lmbda, pltKrIntgd = pltKrIntgd)
    res = quad(intgd, 0, kmax, epsabs=epsabs, epsrel=epsrel)
    _checkIntegral(res, eps = 1e-2, origin = "cmpFr")
    k_int = res[0]
    fr = -2 * np.pi * G * rho * rhop * alpha * k_int
    #fr = -2 * G * rho * rhop * alpha * k_int
    #print('Fr...', fr)

    if plotIntgd:
        print("anaCGrav2.cmpFr: plotting k-integrand")
        k = np.linspace(0, kmax, nk4plot)
        r = np.zeros(nk4plot)
        for i in range(nk4plot):
            if i % 250 == 0:
                print('---> ', i, k[i])
            r[i] = k_intgd_r(k[i], xs, zs, ap, L, a, b, l, lmbda = lmbda)
        trap = trapz(r, x = k)
        print("trapz estimation: ", trap)
        print("quad estimation: ", res)
        plt.plot(k, r)
        plt.ylabel('Fr intgd')
        plt.xlabel('k')
        plt.show()
    
    return fr


def cmpStiffness(_axis, ap, bp, L, rhop, a, b, l, rho, zs = 0, alpha = 1, lmbda = 1, kmax = 'auto'):
    """
    Compute stiffness along cylinders axis between two hollow cylinders
    _axis -- x for radial stiffness, z for longitudinal
    zs -- reference position (along z) of target cylinder. Can only be 0 or |z|>l+L, in which case cylinders are fully outside one of the other (above each other)
    ap, bp -- inner and outer radii of target cylinder
    L -- half-height of target cylinder
    rhop -- density of target cylinder
    a, b -- inner and outer radii of source cylinder
    l -- half-height of source cylinder
    rho -- density of source cylinder
    """
    if _axis == 'x':
        return cmpRadialStiffness(ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, kmax = kmax)
    elif _axis == 'z':
        return cmpLongitudinalStiffness(ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, kmax = kmax)
    else:
        raise ValueError('Bad _axis', _axis)
    

def cmpForceTaylor3(d, _axis, ap, bp, L, rhop, a, b, l, rho, zs = 0, alpha = 1, lmbda = 1, kmax = 'auto'):
    """
    Compute force as a 3rd order Taylor expansion (along cylinders axis between two hollow cylinders)
    zs -- reference position (along z) of target cylinder. Can only be 0 or |z|>l+L, in which case cylinders are fully outside one of the other (above each other)
    ap, bp -- inner and outer radii of target cylinder
    L -- half-height of target cylinder
    rhop -- density of target cylinder
    a, b -- inner and outer radii of source cylinder
    l -- half-height of source cylinder
    rho -- density of source cylinder
    """
    if _axis == 'x':
        return cmpRadialForceTaylor3(d, ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, kmax = kmax)
    elif _axis == 'z':
        return cmpLongitudinalForceTaylor3(d, ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, kmax = kmax)
    else:
        raise ValueError('Bad axis', _axis)
    
def cmpLongitudinalForceTaylor3(d, ap, bp, L, rhop, a, b, l, rho, zs = 0, alpha = 1, lmbda = 1, kmax = 'auto', epsabs=1.49e-13, epsrel=1.49e-13):
    """
    Compute longitudinal force as a 3rd order Taylor expansion (along cylinders axis between two hollow cylinders)
    zs -- reference position (along z) of target cylinder. Can only be 0 or |z|>l+L, in which case cylinders are fully outside one of the other (above each other)
    ap, bp -- inner and outer radii of target cylinder
    L -- half-height of target cylinder
    rhop -- density of target cylinder
    a, b -- inner and outer radii of source cylinder
    l -- half-height of source cylinder
    rho -- density of source cylinder
    """
    if (zs != 0 and np.abs(zs) <= l + L) or (zs == 0 and l == L):
        raise ValueError("Geometry not supported.", zs, l, L)

    if kmax == 'auto':
        if zs == 0 and l > L:
            kmax = 15 / (l - L) #15 adhoc
        elif zs == 0 and l < L:
            kmax = 15 / (L - l) #15 adhoc
        else:
            kmax = 15 / (np.abs(zs) - l - L)
        
    if zs == 0:
        #case A (l > L) or D (l < L)
        intgd1 = lambda k: _kzAD_int(k, a, b, l, ap, bp, L, lmbda)
        _int1 = quad(intgd1, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int1, eps = 1e-2, origin = "cmpLongitudinalForceTaylor3 (A/D #1)")
        intgd3 = lambda k: _k3zAD_int(k, a, b, l, ap, bp, L, lmbda)
        _int3 = quad(intgd3, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int3, eps = 1e-2, origin = "cmpLongitudinalForceTaylor3 (A/D #3)")
        prefac = -16 * np.pi**2 * G * rho * rhop * alpha
        F = prefac * (_int1[0] * d + _int3[0] * d**3)
    else:
        #case B1/B2
        intgd0 = lambda k: _k0zB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int0 = quad(intgd0, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int0, eps = 1e-2, origin = "cmpLongitudinalForceTaylor3 (B #0)")
        intgd1 = lambda k: _k1zB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int1 = quad(intgd1, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int1, eps = 1e-2, origin = "cmpLongitudinalForceTaylor3 (B #1)")
        intgd2 = lambda k: _k2zB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int2 = quad(intgd2, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int2, eps = 1e-2, origin = "cmpLongitudinalForceTaylor3 (B #2)")
        intgd3 = lambda k: _k3zB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int3 = quad(intgd3, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int3, eps = 1e-2, origin = "cmpLongitudinalForceTaylor3 (B #3)")
        prefac = 16 * np.pi**2 * G * rho * rhop * alpha
        F = prefac * (_int0[0] + _int1[0] * d + _int2[0] * d**2 + _int3[0] * d**3)
    return F

def cmpRadialForceTaylor3(d, ap, bp, L, rhop, a, b, l, rho, zs = 0, alpha = 1, lmbda = 1, kmax = 'auto', epsabs=1.49e-13, epsrel=1.49e-13, niterMax = 15, kincr = 2, tol = 1e-3):
    """
    Compute stiffness along cylinders axis between two hollow cylinders
    zs -- reference position (along z) of target cylinder. Can only be 0 or |z|>l+L, in which case cylinders are fully outside one of the other (above each other)
    ap, bp -- inner and outer radii of target cylinder
    L -- half-height of target cylinder
    rhop -- density of target cylinder
    a, b -- inner and outer radii of source cylinder
    l -- half-height of source cylinder
    rho -- density of source cylinder
    """
    if (zs != 0 and np.abs(zs) <= l + L) or (zs == 0 and l == L):
        raise ValueError("Geometry not supported.")

    if kmax == 'auto':
        #print("WARNING! anaCGrav2.cmpRadialStiffness: auto kmax sets it to 500")
        ks, kmax = cmpRadialStiffness(ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, kmax = 'auto', niterMax = niterMax, kincr = kincr, tol = tol, get_kmax = True)
        
    
    if zs == 0:
        #case A (l > L) or D (l < L)
        intgd1 = lambda k: _k1rAD_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int1 = quad(intgd1, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int1, eps = 1e-2, origin = "cmpRadialForceTaylor3 (A/D #1)")
        intgd3 = lambda k: _k3rAD_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int3 = quad(intgd3, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int3, eps = 1e-2, origin = "cmpRadialForceTaylor3 (A/D #3)")
        prefac = -2 * np.pi**2 * G * rho * rhop * alpha
        F = prefac * (_int1[0] * d + _int3[0] * d**3)
    else:
        #case B1/B2
        intgd1 = lambda k: _k1rB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int1 = quad(intgd1, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int1, eps = 1e-2, origin = "cmpRadialForceTaylor3 (B #1)")
        intgd3 = lambda k: _k3rB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int3 = quad(intgd3, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int3, eps = 1e-2, origin = "cmpRadialForceTaylor3 (B #3)")
        prefac = -2 * np.pi**2 * G * rho * rhop * alpha
        F = prefac * (_int1[0] * d + _int3[0] * d**3)
    return F

def cmpLongitudinalStiffness(ap, bp, L, rhop, a, b, l, rho, zs = 0, alpha = 1, lmbda = 1, kmax = 'auto', epsabs=1.49e-13, epsrel=1.49e-13):
    """
    Compute stiffness along cylinders axis between two hollow cylinders
    zs -- reference position (along z) of target cylinder. Can only be 0 or |z|>l+L, in which case cylinders are fully outside one of the other (above each other)
    ap, bp -- inner and outer radii of target cylinder
    L -- half-height of target cylinder
    rhop -- density of target cylinder
    a, b -- inner and outer radii of source cylinder
    l -- half-height of source cylinder
    rho -- density of source cylinder
    """
    if (zs != 0 and np.abs(zs) <= l + L) or (zs == 0 and l == L):
        raise ValueError("Geometry not supported.")

    if kmax == 'auto':
        if zs == 0 and l > L:
            kmax = 15 / (l - L) #15 adhoc
        elif zs == 0 and l < L:
            kmax = 15 / (L - l) #15 adhoc
        else:
            kmax = 15 / (np.abs(zs) - l - L)
        
    if zs == 0:
        #case A (l > L) or D (l < L)
        intgd = lambda k: _kzAD_int(k, a, b, l, ap, bp, L, lmbda)
        _int = quad(intgd, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int, eps = 1e-2, origin = "cmpLongitudinalStiffness (A/D)")
        prefac = -16 * np.pi**2 * G * rho * rhop * alpha
        ky = prefac * _int[0]
    else:
        #case B1/B2
        intgd = lambda k: _kzB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int = quad(intgd, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int, eps = 1e-2, origin = "cmpLongitudinalStiffness (B)")
        prefac = 16 * np.pi**2 * G * rho * rhop * alpha
        ky = prefac * _int[0]

    return ky

def cmpRadialStiffness(ap, bp, L, rhop, a, b, l, rho, zs = 0, alpha = 1, lmbda = 1, kmax = 'auto', epsabs=1.49e-13, epsrel=1.49e-13, niterMax = 15, kincr = 2, tol = 1e-3, get_kmax = False):
    """
    Compute stiffness along cylinders axis between two hollow cylinders
    zs -- reference position (along z) of target cylinder. Can only be 0 or |z|>l+L, in which case cylinders are fully outside one of the other (above each other)
    ap, bp -- inner and outer radii of target cylinder
    L -- half-height of target cylinder
    rhop -- density of target cylinder
    a, b -- inner and outer radii of source cylinder
    l -- half-height of source cylinder
    rho -- density of source cylinder
    """
    if kmax == 'auto':
        #print("WARNING! anaCGrav2.cmpRadialStiffness: auto kmax sets it to 500")
        kmax = 500
        print("--> anaCGrav2.cmpRadialStiffness: trying kmax=" + str(kmax) + "; iter=0")
        k0 = _cmpRadialStiffnessLoop(kmax, ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda)
        print("   k0=" + str(k0))
        iprec = 1
        _it = 1
        kip = k0
        while iprec > tol and _it <= niterMax:
            kmax *= kincr
            print("--> anaCGrav2.cmpRadialStiffness: trying kmax=" + str(kmax) + "; iter=" + str(_it))
            ki = _cmpRadialStiffnessLoop(kmax, ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, epsabs=epsabs, epsrel=epsrel)
            if not np.isnan(ki):
                iprec = np.abs((ki - kip) / kip)
                print("   k=" + str(ki) + ", prec=" + str(iprec))
                kip = ki
                ki2keep = ki
            else:
                print("   WARNING! ki is nan. Trying further...")
            _it += 1
        if iprec > tol:
            print("WARNING! anaCGrav2.cmpRadialStiffness: did not converge after " + str(niterMax) + " iteration (achieved precision = " + str(iprec) + ")")
        if not get_kmax:
            return ki2keep
        else:
            return ki2keep, kmax

    else:
        if not get_kmax:
            return _cmpRadialStiffnessLoop(kmax, ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, epsabs=epsabs, epsrel=epsrel)
        else:
            return _cmpRadialStiffnessLoop(kmax, ap, bp, L, rhop, a, b, l, rho, zs = zs, alpha = alpha, lmbda = lmbda, epsabs=epsabs, epsrel=epsrel), kmax
    
def _cmpRadialStiffnessLoop(kmax, ap, bp, L, rhop, a, b, l, rho, zs = 0, alpha = 1, lmbda = 1, epsabs=1.49e-13, epsrel=1.49e-13):
    """
    Compute stiffness along cylinders axis between two hollow cylinders
    zs -- reference position (along z) of target cylinder. Can only be 0 or |z|>l+L, in which case cylinders are fully outside one of the other (above each other)
    ap, bp -- inner and outer radii of target cylinder
    L -- half-height of target cylinder
    rhop -- density of target cylinder
    a, b -- inner and outer radii of source cylinder
    l -- half-height of source cylinder
    rho -- density of source cylinder
    """
    if (zs != 0 and np.abs(zs) <= l + L) or (zs == 0 and l == L):
        raise ValueError("Geometry not supported.")

    if kmax == 'auto':
        #print("WARNING! anaCGrav2.cmpRadialStiffness: auto kmax sets it to 500")
        kmax = 500
        
    
    if zs == 0:
        #case A (l > L) or D (l < L)
        intgd = lambda k: _krAD_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int = quad(intgd, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int, eps = 1e-2, origin = "_cmpRadialStiffnessLoop (A/D)")
        prefac = -2 * np.pi**2 * G * rho * rhop * alpha
        ky = prefac * _int[0]
    else:
        #case B1/B2
        intgd = lambda k: _krB_int(k, zs, a, b, l, ap, bp, L, lmbda)
        _int = quad(intgd, 0, kmax, epsabs=epsabs, epsrel=epsrel)
        _checkIntegral(_int, eps = 1e-2, origin = "_cmpRadialStiffnessLoop (B)")
        prefac = -2 * np.pi**2 * G * rho * rhop * alpha
        ky = prefac * _int[0]

    return ky


def _kzAD_int(k, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for stiffness along z in A and D cases
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    if l < L:
        num2 = np.exp(-kappa * L) * np.sinh(kappa * l)
    else:
        num2 = np.exp(-kappa * l) * np.sinh(kappa * L)
    denom = kappa * k
    return num1 * num2 / denom

def _kzB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for stiffness along z in B case
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    num2 = np.exp(-kappa * np.abs(zs)) * np.sinh(kappa * l) * np.sinh(kappa * L)
    denom = kappa * k
    return num1 * num2 / denom

def _k3zAD_int(k, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 3rd order force along z in A and D cases
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = kappa * (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    if l < L:
        num2 = np.exp(-kappa * L) * np.sinh(kappa * l)
    else:
        num2 = np.exp(-kappa * l) * np.sinh(kappa * L)
    denom = 6 * k
    return num1 * num2 / denom

def _k0zB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 0th order force along z in B case
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    num2 = np.exp(-kappa * np.abs(zs)) * np.sinh(kappa * l) * np.sinh(kappa * L)
    denom = kappa**2 * k
    return -num1 * num2 / denom * np.sign(zs)

def _k1zB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 1st order force along z in B case
    """
    return _kzB_int(k, zs, a, b, l, ap, bp, L, lmbda)

def _k2zB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 2nd order force along z in B case
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    num2 = np.exp(-kappa * np.abs(zs)) * np.sinh(kappa * l) * np.sinh(kappa * L)
    denom = 2 * k
    return -num1 * num2 / denom * np.sign(zs)

def _k3zB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 3rd order force along z in B case
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = kappa * (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    num2 = np.exp(-kappa * np.abs(zs)) * np.sinh(kappa * l) * np.sinh(kappa * L)
    denom = 6 * k
    return num1 * num2 / denom
    
def _krAD_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for stiffness along x in A and D cases
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = k * (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    #if l > L:
    #    num2 = 4 * L - 4 * np.exp(-kappa * l) / kappa * np.sinh(kappa * L) * np.cosh(kappa * zs)
    #else:
    #    num2 = 4 * l - 4 * np.exp(-kappa * L) / kappa * np.sinh(kappa * l) * np.cosh(kappa * zs)
    num2 = H1(l,L,zs, kappa) + H4(l,L,zs, kappa) - H3(l,L,zs, kappa)
    return num1 * num2 / kappa**2

def _k1rAD_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 1st order force along x in A and D cases
    """
    return _krAD_int(k, zs, a, b, l, ap, bp, L, lmbda)

def _k3rAD_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 3rd order force along x in A and D cases
    """
    kappa = np.sqrt(k**2 + 1./lmbda**2)
    num1 = -1./4 * k**3 * (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    #if l > L:
    #    num2 = 4 * L - 4 * np.exp(-kappa * l) / kappa * np.sinh(kappa * L) * np.cosh(kappa * zs)
    #else:
    #    num2 = 4 * l - 4 * np.exp(-kappa * L) / kappa * np.sinh(kappa * l) * np.cosh(kappa * zs)
    num2 = H1(l,L,zs, kappa) + H4(l,L,zs, kappa) - H3(l,L,zs, kappa)
    return num1 * num2 / kappa**2

def _krB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for stiffness along x in B case. same implementation as A/D cases
    """
    return _krAD_int(k, zs, a, b, l, ap, bp, L, lmbda)
    #kappa = np.sqrt(k**2 + 1./lmbda**2)
    #num1 = k * (bp * j1(k * bp) - ap * j1(k * ap)) * (b * j1(k * b) - a * j1(k * a))
    ##num2 = 4 * np.exp(-kappa * np.abs(zs)) / kappa * np.sinh(kappa * l) * np.sinh(kappa * L)
    #num2 = H1(l,L,zs, kappa) + H4(l,L,zs, kappa) - H3(l,L,zs, kappa)
    #return num1 * num2 / kappa**2

def _k1rB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 1st order force along x in B case
    """
    return _krB_int(k, zs, a, b, l, ap, bp, L, lmbda)

def _k3rB_int(k, zs, a, b, l, ap, bp, L, lmbda):
    """
    Integrand for 3rd order force along x in B case
    The implementation is the same as for A/D cases
    """
    return _k3rAD_int(k, zs, a, b, l, ap, bp, L, lmbda)
    #kappa = np.sqrt(k**2 + 1./lmbda**2)
    #num1 = -4 * k**2 * (2 * (j0(k * bp) - j0(k * ap)) + k * (bp * j1(k * bp) - ap * j1(k * ap))) * (b * j1(k * b) - a * j1(k * a))
    #num2 = 4 * np.exp(-kappa * np.abs(zs)) / kappa * np.sinh(kappa * l) * np.sinh(kappa * L)
    #return num1 * num2 / kappa**2


def _checkIntegral(_int, eps = 1e-2, origin = "_checkIntegral"):
    """
    Check output of quad
    """
    if _int[1] / np.abs(_int[0]) > eps:
        print("WARNING! anaCGrav2." + origin + ": quad integral precision pretty bad: integral=" + str(_int[0]) + " -- error=" + str(_int[1]))
        

def tstCase():
    print(getCase(2, 1, 0.1)) #case A
    print(getCase(2, 1, 4)) #case B1
    print(getCase(2, 1, -4)) #case B2
    print(getCase(2, 1, 1.3)) #case C1
    print(getCase(2, 1, -1.3)) #case C2
    print(getCase(2, 5, 0)) #case D
    

def tstH():
    kappa = 1
    l = 2 
    L = 1
    zs = 0.1
    print(getCase(l,L,zs), H1(l,L,zs, kappa), H2(l,L,zs, kappa), H3(l,L,zs, kappa), H4(l,L,zs, kappa))
    l = 2 
    L = 1
    zs = 4
    print(getCase(l,L,zs), H1(l,L,zs, kappa), H2(l,L,zs, kappa), H3(l,L,zs, kappa), H4(l,L,zs, kappa))
    l = 2 
    L = 1
    zs = -4
    print(getCase(l,L,zs), H1(l,L,zs, kappa), H2(l,L,zs, kappa), H3(l,L,zs, kappa), H4(l,L,zs, kappa))
    l = 2 
    L = 1
    zs = 1.3
    print(getCase(l,L,zs), H1(l,L,zs, kappa), H2(l,L,zs, kappa), H3(l,L,zs, kappa), H4(l,L,zs, kappa))
    l = 2 
    L = 1
    zs = -1.3
    print(getCase(l,L,zs), H1(l,L,zs, kappa), H2(l,L,zs, kappa), H3(l,L,zs, kappa), H4(l,L,zs, kappa))
    l = 2 
    L = 5
    zs = 0.1
    print(getCase(l,L,zs), H1(l,L,zs, kappa), H2(l,L,zs, kappa), H3(l,L,zs, kappa), H4(l,L,zs, kappa))


def tstK(a = 1, k = 1, pltIntgd = True):
    trueRes = np.pi * a**2
    xs = a / 5
    K = Ksurf(a, xs)
    k_z = Kz(k, a, xs, pltIntgd = pltIntgd)
    k_z2 = Kz2D(k, a, xs, pltIntgd = pltIntgd)
    k_r = Kr(k, a, xs, pltIntgd = pltIntgd)
    print(a, xs, '--', trueRes, K)
    print('   ', k_z, k_z2, k_r)
    print('---------------------')
    
    xs = -a / 5
    K = Ksurf(a, xs)
    k_z = Kz(k, a, xs, pltIntgd = pltIntgd)
    k_z2 = Kz2D(k, a, xs, pltIntgd = pltIntgd)
    k_r = Kr(k, a, xs, pltIntgd = pltIntgd)
    print(a, xs, '--', trueRes, K)
    print('   ', k_z, k_z2, k_r)
    print('---------------------')

    xs = 0
    K = Ksurf(a, xs)
    k_z = Kz(k, a, xs, pltIntgd = pltIntgd)
    k_z2 = Kz2D(k, a, xs, pltIntgd = pltIntgd)
    k_r = Kr(k, a, xs, pltIntgd = pltIntgd)
    print(a, xs, '--', trueRes, K)
    print('   ', k_z, k_z2, k_r)
    print('---------------------')
    
    xs = 3 * a
    K = Ksurf(a, xs)
    k_z = Kz(k, a, xs, pltIntgd = pltIntgd)
    k_z2 = Kz2D(k, a, xs, pltIntgd = pltIntgd)
    k_r = Kr(k, a, xs, pltIntgd = pltIntgd)
    print(a, xs, '--', trueRes, K)
    print('   ', k_z, k_z2, k_r)
    print('---------------------')
    
    xs = -3 * a
    K = Ksurf(a, xs)
    k_z = Kz(k, a, xs, pltIntgd = pltIntgd)
    k_z2 = Kz2D(k, a, xs, pltIntgd = pltIntgd)
    k_r = Kr(k, a, xs, pltIntgd = pltIntgd)
    print(a, xs, '--', trueRes, K)
    print('   ', k_z, k_z2, k_r)
    print('---------------------')

    
def tst_kintgd(kmax = 500, nk = 50000, xs = 1, zs = 0.1, ap = 0.5, L = 1, a = 0.4, b = 0.5, l = 2, lmbda = 1):
    xs = 0.1
    zs = 0.0
    ap = 0.07
    L = 0.15
    a = 0
    b = 0.13
    l = 0.15

    k = np.linspace(0, kmax, nk)
    z = np.zeros(nk)
    r = np.zeros(nk)
    for i in range(nk):
        if i % 250 == 0:
            print('---> ', i, k[i])
        z[i] = k_intgd_z(k[i], xs, zs, ap, L, a, b, l, lmbda = lmbda)
        r[i] = k_intgd_r(k[i], xs, zs, ap, L, a, b, l, lmbda = lmbda)
    Fr = cmpFr(xs, zs, ap, L, 1, a, b, l, 1, alpha = 1, lmbda = lmbda, kmax = kmax)
    print('Fr:', Fr)
    ax = plt.subplot(211)
    plt.plot(k, z)
    plt.ylabel('Fz intgd')
    plt.subplot(212, sharex = ax)
    plt.plot(k, r)
    plt.ylabel('Fr intgd')
    plt.xlabel('k')
    plt.show()


def tst_intgds(_axis, delta = None, xs = 0, zs = 0., cset = None, ap = 0.07, bp = 0.1, L = 0.11, a = 0.13, b = 0.16, l = 0.15, rho = 2000, rhop = 2000, alpha = 1, kmax = 500, nk = 50000, lmbda = 1., zoom = False):
    if delta is None:
        delta = [-0.01, -0.007, -0.005, -0.002, -0.0005, -0.0002, 0.0001, 0.0006, 0.001, 0.003, 0.006, 0.008, 0.01]
        if zoom:
            delta = np.array(delta) / 10
    if cset == 1210:
        bp = 7e-2
        ap = 0#.000001
        L = 15e-2
        b = 13e-2
        a = 0#.000001
        l = 15e-2
        rhop = 2000
        rho = 2000
        xs = 0
        zs = 0.35
    elif cset == 1220:
        bp = 7e-2
        ap = 0#.000001
        L = 15e-2
        b = 13e-2
        a = 0#.000001
        l = 15e-2
        rhop = 2000
        rho = 2000
        xs = 0
        zs = -0.35
    elif cset == 211:
        bp = 7e-2
        ap = 3e-2
        L = 15e-2
        b = 13e-2
        a = 10e-2
        l = 5e-2
        rhop = 2000
        rho = 2000
        xs = 0
        zs = 0
    elif cset == 212:
        bp = 7e-2
        ap = 3e-2
        L = 5e-2
        b = 13e-2
        a = 10e-2
        l = 15e-2
        rhop = 2000
        rho = 2000
        xs = 0
        zs = 0
    elif cset == 311:
        bp = 7e-2
        ap = 6.5e-2
        L = 15e-2
        b = 2e-2
        a = 1.6e-2
        l = 5e-2
        rhop = 2000
        rho = 2000
        xs = 0
        zs = 0
    elif cset == 312:
        bp = 7e-2
        ap = 6.5e-2
        L = 5e-2
        b = 2e-2
        a = 1.6e-2
        l = 15e-2
        rhop = 2000
        rho = 2000
        xs = 0
        zs = 0
    elif cset is not None:
        raise ValueError('Bad cset')
    else:
        pass
    l *= 0.5 #l and L in functions below are half-heights
    L *= 0.5
        
    nd = np.size(delta)
    Fana = np.zeros(nd)
    Fk = np.zeros(nd)
    for i in range(nd):
        if cset is None:
            fname = 'Figure_' + _axis + '_' + str(delta[i]) + '.png'
        else:
            fname = 'Figure_' + _axis + '_' + str(cset) + '_' + str(delta[i]) + '.png'
        Fana[i], Fk[i], ky = tst_intgds1(_axis, delta = delta[i], xs = xs, zs = zs, ap = ap, bp = bp, L = L, a = a, b = b, l = l, rho = rho, rhop = rhop, alpha = alpha, kmax = kmax, nk = nk, lmbda = lmbda, fname = fname, xplot = False)
    kana = (Fana[-1] - Fana[0]) / (delta[-1] - delta[0])
    print("---> stiffness (ana, direct)", kana, ky)
    plt.plot(delta, Fana, label = 'ana; k=' + str(kana))
    plt.plot(delta, Fk, label = 'stiff; k=' + str(ky))
    plt.legend()
    plt.xlabel('delta')
    plt.ylabel('F')
    plt.show()
    
def tst_intgds1(_axis, delta = 0.01, xs = 0, zs = 0., ap = 0.07, bp = 0.1, L = 0.11, a = 0.13, b = 0.16, l = 0.15, rho = 2000, rhop = 2000, alpha = 1, kmax = 500, nk = 50000, lmbda = 1, fname = 'Fig.png', xplot = True):
    if _axis == 'z' and zs != 0:
        raise NotImplementedError("This is too much of a pain in the ass... But seems to work, so don't bother with it.")
    
    if _axis == 'x':
        pref_ana = -2 * np.pi * G * rho * rhop * alpha
        pref_stiff = -8 * np.pi * G * rho * rhop * alpha
    elif _axis == 'z':
        pref_ana = -2 * np.pi * G * rho * rhop * alpha
        pref_stiff = -16 * np.pi**2 * G * rho * rhop * alpha
    else:
        raise ValueError('Fuck you!!!')
    
    k = np.linspace(0, kmax, nk)
    ana = np.zeros(nk)
    stiff = np.zeros(nk)
    for i in range(nk):
        if i % 250 == 0:
            print('---> ', i, k[i])
        if _axis == 'x':
            if zs < 0.5 * min(l,L) and np.abs(zs) < l + L:
                stiff[i] = _krAD_int(k[i], zs, a, b, l, ap, bp, L, lmbda)
            else:
                if np.abs(zs) < l + L:
                    raise ValueError("Not good... Need |zs| > l + L if zs != 0")
                stiff[i] = _krB_int(k[i], zs, a, b, l, ap, bp, L, lmbda)
            ana[i] = k_intgd_r(k[i], xs + delta, zs, bp, L, a, b, l, lmbda = lmbda) - k_intgd_r(k[i], xs + delta, zs, ap, L, a, b, l, lmbda = lmbda)
        elif _axis == 'z':
            if zs == 0:
                stiff[i] = _kzAD_int(k[i], a, b, l, ap, bp, L, lmbda)
            else:
                raise NotImplementedError("Not necessary. See pain in the arse above")
            ana[i] = k_intgd_z(k[i], xs, zs + delta, bp, L, a, b, l, lmbda = lmbda) - k_intgd_z(k[i], xs, zs + delta, ap, L, a, b, l, lmbda = lmbda)
        else:
            raise ValueError('Fuck you!!!')


    ana *= pref_ana
    stiff *= pref_stiff

    #tricky: xs/zs is in ana, but not in stiff. So to compare them, need to mutliply stiff by xs/zs
    if _axis == 'x':
        stiff *= delta
    else:
        stiff *= delta
        
    if _axis == 'x':
        Fana = cmpFr(xs + delta, zs, bp, L, rhop, a, b, l, rho, alpha = alpha, lmbda = lmbda, kmax = kmax) - cmpFr(xs + delta, zs, ap, L, rhop, a, b, l, rho, alpha = alpha, lmbda = lmbda, kmax = kmax)
        if zs < 0.5 * min(l,L) and np.abs(zs) < l + L:
            intgd1 = lambda k: _krAD_int(k, zs, a, b, l, ap, bp, L, lmbda)
            _int = quad(intgd1, 0, kmax)
            Fstiff = pref_stiff * _int[0] * delta
        else:
            intgd1 = lambda k: _krB_int(k, zs, a, b, l, ap, bp, L, lmbda)
            _int = quad(intgd1, 0, kmax)
            Fstiff = pref_stiff * _int[0] * delta
    elif _axis == 'z':
        Fana = cmpFz(xs, zs + delta, bp, L, rhop, a, b, l, rho, alpha = alpha, lmbda = lmbda, kmax = kmax) - cmpFz(xs, zs + delta, ap, L, rhop, a, b, l, rho, alpha = alpha, lmbda = lmbda, kmax = kmax)
        if zs == 0:
            intgd1 = lambda k: _kzAD_int(k, a, b, l, ap, bp, L, lmbda)
            _int = quad(intgd1, 0, kmax)
            Fstiff = pref_stiff * _int[0] * delta
        else:
            raise NotImplementedError("Not necessary. See pain in the arse above")
    else:
        raise ValueError('Fuck you!!!')
    ky = pref_stiff * _int[0]

    print('Fana, Fstiff', Fana, Fstiff)
    
    #ax = plt.subplot(311)
    #plt.plot(k, ana)
    #plt.ylabel('F intgd')
    #plt.subplot(312, sharex = ax)
    #plt.plot(k, stiff)
    #plt.ylabel('Stiffness intgd')
    #plt.subplot(313, sharex = ax)
    plt.plot(k, ana, label = 'ana; F=' + str(Fana))
    plt.plot(k, stiff, label = 'stiff; F=' + str(Fstiff))
    plt.legend()
    plt.ylabel('k-integrand')
    plt.xlabel('k')
    plt.suptitle('delta=' + str(delta))
    if xplot:
        plt.show()
    else:
        plt.savefig(fname)
        plt.cla()
        plt.clf()

    return Fana, Fstiff, ky

def tstF(xs = 1, zs = 0.1, ap = 0.5, L = 1, rhop = 1, a = 0.4, b = 0.5, l = 2, rho = 1, alpha = 1, lmbda = 1, kmax = 50):
    t0 = time()
    fz = cmpFz(xs, zs, ap, L, rhop, a, b, l, rho, alpha = alpha, lmbda = lmbda, kmax = kmax)
    t1 = time()
    print('---> Fz', t1 - t0)
    print('  ', fz)
    fr = cmpFr(xs, zs, ap, L, rhop, a, b, l, rho, alpha = alpha, lmbda = lmbda, kmax = kmax)
    t2 = time()
    print('---> Fr', t2 - t1)
    print('  ', fr)
    

def tstStruve_k(k, a = 1, xs = 3):
    intgd = lambda r: k * r * j1(k * r)
    if np.abs(xs) > a:
        theta_min = -np.arcsin(a / np.abs(xs)) * 0.999999 #0.99999 to avoid 0 in sqrt in integrand at the boundaries of integration
        theta_max = np.arcsin(a / np.abs(xs)) * 0.999999
        if xs < 0:
            theta_min += np.pi
            theta_max += np.pi
    else:
        theta_min = 0
        theta_max = 2. * np.pi

    theta = np.linspace(theta_min, theta_max, 500)
    #plt.plot(th, Rint(th))
    #plt.xlabel('theta')
    #plt.ylabel('Kr intgd, k = ' + str(k))
    #plt.show()

    ana = np.pi / 2 * ( (Rplus(theta, a, xs) * j1(k * Rplus(theta, a, xs)) * struve(0, k * Rplus(theta, a, xs)) - Rplus(theta, a, xs) * j0(k * Rplus(theta, a, xs)) * struve(1, k * Rplus(theta, a, xs))) - (Rminus(theta, a, xs) * j1(k * Rminus(theta, a, xs)) * struve(0, k * Rminus(theta, a, xs)) - Rminus(theta, a, xs) * j0(k * Rminus(theta, a, xs)) * struve(1, k * Rminus(theta, a, xs))))

    num = np.zeros(np.size(theta))
    for i in range(np.size(theta)):
        res = quad(intgd, Rminus(theta[i], a, xs), Rplus(theta[i], a, xs))
        num[i] = res[0]

    ax = plt.subplot(211)
    plt.plot(theta, ana, label = 'ana')
    plt.plot(theta, num, label = 'num')
    plt.xlabel('theta')
    plt.legend()
    plt.subplot(212, sharex = ax)
    plt.plot(theta, ana - num)
    plt.xlabel('theta')
    plt.ylabel('ana - num')
    plt.suptitle('k='+str(k))
    plt.show()

def tstStruve(a = 1, xs = 3, k = np.linspace(0, 500, 15)):
    for ik in k:
        tstStruve_k(ik, a = a, xs = xs)
        
