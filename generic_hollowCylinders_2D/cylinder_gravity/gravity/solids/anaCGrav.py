import numpy as np
from scipy.integrate import quad, dblquad
#from gravity.solids.analCyls import anaIxz as analIxz
from gravity.solids.analCyls import *
import matplotlib.pyplot as plt

#a1 = 1.1
#a2 = 1.4
#t = 3.1

## potential definition
# Newton
def V(x, y, z):
    return 1./np.sqrt(x**2 + y**2 + z**2)

def dVx(x1, x2, y, z):
    x = x1 - x2
    return -x / (x**2 + y**2 + z**2)**(3./2)

def Vz(z, y, x):
    return 1./np.sqrt(x**2 + y**2 + z**2)

def dVz(z1, z2, y, x):
    z = z1 - z2
    return -z / (x**2 + y**2 + z**2)**(3./2)

# Yukawa
def VYuk(x, y, z, lmbda):
    return 1./np.sqrt(x**2 + y**2 + z**2) * np.exp(-np.sqrt(x**2 + y**2 + z**2) / lmbda)

def VzYuk(z, y, x, lmbda):
    return 1./np.sqrt(x**2 + y**2 + z**2) * np.exp(-np.sqrt(x**2 + y**2 + z**2) / lmbda)


########################
####### Fx #############
########################
## integral over x1 and x2 (Ix)
def numIx_x2(y1_in, y2_in, z, a1, a2, t):
    """
    2D numerical integration of V(x1, x2, y, z).dx1.dx2
    """
    #dv = lambda x1, x2: dVx(x1, x2, y, z)
    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)
    x1min = x1min_fun(a1, y1)
    x1max = x1max_fun(a1, y1)
    if n > 1:
        res = np.zeros(n)
        for i in range(n):
            resi, err = dblquad(dVx, x1min[i], x1max[i], x2min_fun(a2, y2[i], t), x2max_fun(a2, y2[i], t), args = (y[i], z))
            res[i] = resi
    else:
        res, err = dblquad(dVx, x1min, x1max, x2min_fun(a2, y2, t), x2max_fun(a2, y2, t), args = (y, z))
    return res

def numIx(y1_in, y2_in, z, a1, a2, t, yukawa = False, lmbda = None):
    """
    1D numerical integration of V(x1, x2, y, z).dx1.dx2
    """
    #vf = lambda x: V(x, y1 - y2, z)
    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)
    
    t0 = t0_fun(y1, y2, a1, a2, t)
    t1 = t1_fun(y1, y2, a1, a2, t)
    t2 = t2_fun(y1, y2, a1, a2, t)
    t3 = t3_fun(y1, y2, a1, a2, t)
    if not yukawa:
        if n > 1:
            res1 = np.zeros(n)
            res2 = np.zeros(n)
            for i in range(n):
                res1[i], err1 = quad(V, t3[i], t2[i], args = (y[i], z))
                res2[i], err2 = quad(V, t0[i], t1[i], args = (y[i], z))
        else:
            res1, err1 = quad(V, t3, t2, args = (y, z))
            res2, err2 = quad(V, t0, t1, args = (y, z))
    else:
        if lmbda is None:
            raise TypeError("lmbda must be set to work with Yukawa")
        if n > 1:
            res1 = np.zeros(n)
            res2 = np.zeros(n)
            for i in range(n):
                res1[i], err1 = quad(VYuk, t3[i], t2[i], args = (y[i], z, lmbda))
                res2[i], err2 = quad(VYuk, t0[i], t1[i], args = (y[i], z, lmbda))
        else:
            res1, err1 = quad(VYuk, t3, t2, args = (y, z, lmbda))
            res2, err2 = quad(VYuk, t0, t1, args = (y, z, lmbda))
    return res1 - res2


def anaIx(y1, y2, z, a1, a2, t, x0 = 1, version = 1):
    """
    Analytic integration for Ix
    x0 is artificial to have dimensionless quantities inside logs and should not affect the result
    """
    def cmpi(i):
        ti = ti_fun(i, y1, y2, a1, a2, t)
        if version == 1:
            return (-1)**i * np.log((np.sqrt(ti**2 + y**2 + z**2) + ti) / x0)
        elif version == 2:
            #if y**2 + z**2 > 0:
            num = np.sqrt(ti**2 + y**2 + z**2) + np.abs(ti)
            denom = np.sqrt(y**2 + z**2)
            #else:
            #    return (-1)**i * np.log((np.sqrt(ti**2 + y**2 + z**2) + ti) / x0)
            return (-1)**i * np.sign(ti) * np.log(num / denom)
        else:
            raise ValueError()

    y = y1 - y2
    res = cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)
    return res


## integral over z (Ixz)
def numIx_x2_z1z2(z1, z2, y1, y2, a1, a2, t):
    return numIx_x2(y1, y2, z1 - z2, a1, a2, t)

def numIxz_z2(y1_in, y2_in, a1, a2, t, z1min, z1max, z2min, z2max):
    """
    2D numerical integration of Ix(y1, y2, z1, z2).dz1.dz2
    To be consistent, use 2D numerical integration of V(x,y,z) to get Ix
    """
    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)
    #Ix = numIx_x2(y1, y2, z, a1, a2, t)
    if n > 1:
        res = np.zeros(n)
        for i in range(n):
            resi, err = dblquad(numIx_x2_z1z2, z1min, z1max, z2min, z2max, args = (y1[i], y2[i], a1, a2, t))
            
            res[i] = resi
    else:
        res, err = dblquad(numIx_x2_z1z2, z1min, z1max, z2min, z2max, args = (y1, y2, a1, a2, t))
    return res


def numIxz(y1_in, y2_in, a1, a2, t, z1min, z1max, z2min, z2max, yukawa = False, lmbda = None):
    """
    2D numerical integration of Ix(y1, y2, z1, z2).dz1.dz2
    To be consistent, use 2D numerical integration of V(x,y,z) to get Ix
    """
    s0 = s0_fun(z1min, z1max, z2min, z2max)
    s1 = s1_fun(z1min, z1max, z2min, z2max)
    s2 = s2_fun(z1min, z1max, z2min, z2max)
    s3 = s3_fun(z1min, z1max, z2min, z2max)

    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)
    #Ix = numIx_x2(y1, y2, z, a1, a2, t)
    if n > 1:
        res = np.zeros(n)
        for i in range(n):
            i_term1 = numIrond(y1[i], y2[i], a1, a2, t, _min = s0, _max = s1, yukawa = yukawa, lmbda = lmbda)
            i_term2 = -numIrond(y1[i], y2[i], a1, a2, t, _min = s3, _max = s2, yukawa = yukawa, lmbda = lmbda)
            j_term1 = -s0 * numJrond(y1[i], y2[i], a1, a2, t, _min = s0, _max = s1, yukawa = yukawa, lmbda = lmbda)
            j_term2 = (s1 - s0) * numJrond(y1[i], y2[i], a1, a2, t, _min = s1, _max = s3, yukawa = yukawa, lmbda = lmbda)
            j_term3 = s2 * numJrond(y1[i], y2[i], a1, a2, t, _min = s3, _max = s2, yukawa = yukawa, lmbda = lmbda)
            res[i] = i_term1 + i_term2 + j_term1 + j_term2 +j_term3
    else:
        i_term1 = numIrond(y1, y2, a1, a2, t, _min = s0, _max = s1, yukawa = yukawa, lmbda = lmbda)
        i_term2 = -numIrond(y1, y2, a1, a2, t, _min = s3, _max = s2, yukawa = yukawa, lmbda = lmbda)
        j_term1 = -s0 * numJrond(y1, y2, a1, a2, t, _min = s0, _max = s1, yukawa = yukawa, lmbda = lmbda)
        j_term2 = (s1 - s0) * numJrond(y1, y2, a1, a2, t, _min = s1, _max = s3, yukawa = yukawa, lmbda = lmbda)
        j_term3 = s2 * numJrond(y1, y2, a1, a2, t, _min = s3, _max = s2, yukawa = yukawa, lmbda = lmbda)
        res = i_term1 + i_term2 + j_term1 + j_term2 +j_term3
    
    return res

#def numIx_z(z, y1, y2, a1, a2, t):
#    return numIx(y1, y2, z, a1, a2, t)

def numIrond(y1, y2, a1, a2, t, _min = None, _max = None, yukawa = False, lmbda = None):
    """
    1D numerical integration of z*Ix(z).dz that enters in Ixz
    For consistency, use 1D numerical integration of Ix
    """
    int_z = lambda z: z * numIx(y1, y2, z, a1, a2, t, yukawa = yukawa, lmbda = lmbda)
    if _min is None:
        raise NotImplementedError()
    if _max is None:
        raise NotImplementedError()
    res, err = quad(int_z, _min, _max)
    return res

def numJrond(y1, y2, a1, a2, t, _min = None, _max = None, yukawa = False, lmbda = None):
    """
    Numerical integration of Ix(z).dz that enters in Ixz
    For consistency, use 1D numerical integration of Ix
    """
    int_z = lambda z: numIx(y1, y2, z, a1, a2, t, yukawa = yukawa, lmbda = lmbda)
    if _min is None:
        raise NotImplementedError()
    if _max is None:
        raise NotImplementedError()
    res, err = quad(int_z, _min, _max)
    return res



### a lot of tests for when I was going nuts, but not depressed yet, not very interesting, so better off commented out
# def anaIrond(y1, y2, _min = None, _max = None, x0 = 1):
#     """
#     Analytic integration of z*Ix(z).dz that enters in Ixz
#     """
#     def cmpi(i):
#         ti = ti_fun(i, y1, y2)
#         f1 = 2 * ti * (np.sqrt(ti**2 + y**2 + _max**2) - np.sqrt(ti**2 + y**2 + _min**2))
#         f2 = 2 * _max**2 * np.log((np.sqrt(ti**2 + y**2 + _max**2) + ti) / x0) - 2 * _min**2 * np.log((np.sqrt(ti**2 + y**2 + _min**2) + ti) / x0)
#         f3 = _min**2 - _max**2
#         #f4 = y**2 * np.log( (ti + np.sqrt(ti**2 + y**2 + _max**2)) / (ti - np.sqrt(ti**2 + y**2 + _max**2)) ) - y**2 * np.log( (ti + np.sqrt(ti**2 + y**2 + _min**2)) / (ti - np.sqrt(ti**2 + y**2 + _min**2)) )
#         f4_numlog = (ti + np.sqrt(ti**2 + y**2 + _max**2)) * (ti - np.sqrt(ti**2 + y**2 + _min**2))
#         f4_denomlog = (ti - np.sqrt(ti**2 + y**2 + _max**2)) * (ti + np.sqrt(ti**2 + y**2 + _min**2))
#         f4 = y**2 * np.log(f4_numlog / f4_denomlog)
#         f5 = y**2 * np.log((_max**2 + y**2) / (_min**2 + y**2))
#         #print(f1, f2, f3, f4, f5, f1 + f2 + f3 + f4 + f5)
#         return (-1)**i * (f1 + f2 + f3 + f4 + f5)
    
#     y = y1 - y2
#     #t0 = t0_fun(y1, y2)
#     #t1 = t1_fun(y1, y2)
#     #t2 = t2_fun(y1, y2)
#     #t3 = t3_fun(y1, y2)
#     res = (cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)) / 4
#     return res
    

# def anaJrond(y1, y2, _min = None, _max = None, x0 = 1):
#     """
#     Analytic integration of Ix(z).dz that enters in Ixz
#     """
#     def cmpi(i):
#         ti = ti_fun(i, y1, y2)
#         f1 = _max * np.log( (np.sqrt(ti**2 + y**2 + _max**2) + ti) / x0 ) - _min * np.log( (np.sqrt(ti**2 + y**2 + _min**2) + ti) / x0 )
#         f2 = ti * (np.log( (np.sqrt(ti**2 + y**2 + _max**2) + _max) / x0 ) - np.log( (np.sqrt(ti**2 + y**2 + _min**2) + _min) / x0 ))
#         f3 = -y * np.arctan2(ti * _max, y * np.sqrt(ti**2 + y**2 + _max**2)) + y * np.arctan2(ti * _min, y * np.sqrt(ti**2 + y**2 + _min**2))
#         #f4 = y * (np.arctan2(_max, y) - np.arctan2(_min, y))
#         #f5 = _min - _max
#         f4 = 0
#         f5 = 0
#         return (-1)**i * (f1 + f2 + f3 + f4 + f5)
    
#     y = y1 - y2
#     #t0 = t0_fun(y1, y2)
#     #t1 = t1_fun(y1, y2)
#     #t2 = t2_fun(y1, y2)
#     #t3 = t3_fun(y1, y2)
#     res = cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)
#     return res


# def numIrond_terms(y1, y2, z1min, z1max, z2min, z2max):
#     s0 = s0_fun(z1min, z1max, z2min, z2max)
#     s1 = s1_fun(z1min, z1max, z2min, z2max)
#     s2 = s2_fun(z1min, z1max, z2min, z2max)
#     s3 = s3_fun(z1min, z1max, z2min, z2max)
#     term1 = numIrond(y1, y2, _min = s0, _max = s1)
#     term2 = numIrond(y1, y2, _min = s3, _max = s2)
#     return term1 - term2

# def anaIrond_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = 1):
#     def cmpij(i, j):
#         rij = rij_fun(i, j, y1, y2, z1min, z1max, z2min, z2max)
#         ti = ti_fun(i, y1, y2)
#         sj = si_fun(j, z1min, z1max, z2min, z2max)
#         f1 = ti * rij
#         f2 = (sj**2 + y**2) * np.log((rij + ti) / x0)
#         return (-1)**(i+j+1) * (f1 + f2)

#     y = y1 - y2
#     fac1 = 0
#     for i in range(4):
#         for j in range(4):
#             fac1 += cmpij(i, j)
#     fac1 /= 2
#     return fac1

# def anaIrond_terms_v1(y1, y2, z1min, z1max, z2min, z2max, x0 = 1, v2 = False):
#     def cmpij(i, j):
#         rij = rij_fun(i, j, y1, y2, z1min, z1max, z2min, z2max)
#         ti = ti_fun(i, y1, y2)
#         sj = si_fun(j, z1min, z1max, z2min, z2max)
#         f1 = ti * rij
#         f2 = sj**2 * np.log((rij + ti) / x0)
#         if not v2:
#             f3 = -sj**2
#             f4 = y**2 * np.log(sj**2 + y**2) / 2
#         else:
#             f3 = 0
#             f4 = 0
#         return (-1)**(i+j+1) * (f1 + f2 + f3 + f4)

#     def cmpLog_i(i):
#         ti = ti_fun(i, y1, y2)
#         v = 1
#         for j in range(4):
#             rij = rij_fun(i, j, y1, y2, z1min, z1max, z2min, z2max)
#             v *= (ti - (-1)**j * rij) / (ti + (-1)**j * rij)
#         return (-1)**i * y**2 * np.log(v)
        
#     y = y1 - y2
#     fac1 = 0
#     for i in range(4):
#         for j in range(4):
#             fac1 += cmpij(i, j)
#     fac1 /= 2
#     fac2 = (cmpLog_i(0) + cmpLog_i(1) + cmpLog_i(2) + cmpLog_i(3)) / 4
#     return fac1 + fac2


# def numJrond_terms(y1, y2, z1min, z1max, z2min, z2max):
#     s0 = s0_fun(z1min, z1max, z2min, z2max)
#     s1 = s1_fun(z1min, z1max, z2min, z2max)
#     s2 = s2_fun(z1min, z1max, z2min, z2max)
#     s3 = s3_fun(z1min, z1max, z2min, z2max)
#     term1 = -s0 * numJrond(y1, y2, _min = s0, _max = s1)
#     term2 = (s1 - s0) * numJrond(y1, y2, _min = s1, _max = s3)
#     term3 = s2 * numJrond(y1, y2, _min = s3, _max = s2)
#     return term1 + term2 + term3

# def anaJrond_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = 1, v2 = False):
#     def cmpij(i, j):
#         rij = rij_fun(i, j, y1, y2, z1min, z1max, z2min, z2max)
#         ti = ti_fun(i, y1, y2)
#         sj = si_fun(j, z1min, z1max, z2min, z2max)
#         f1 = sj**2 * np.log((rij + ti) / x0)
#         f2 = ti * sj * np.log((rij + sj) / x0)
#         f3 = -y * sj * np.arctan2(ti * sj, y * rij)
#         return (-1)**(i+j) * (f1 + f2 + f3)
        
#     y = y1 - y2
#     res = 0
#     for i in range(4):
#         for j in range(4):
#             res += cmpij(i, j)

#     if v2:
#         s0 = s0_fun(z1min, z1max, z2min, z2max)
#         s1 = s1_fun(z1min, z1max, z2min, z2max)
#         s2 = s2_fun(z1min, z1max, z2min, z2max)
#         s3 = s3_fun(z1min, z1max, z2min, z2max)
#         term1 = -s0 * anaJrond(y1, y2, _min = s0, _max = s1)
#         term2 = (s1 - s0) * anaJrond(y1, y2, _min = s1, _max = s3)
#         term3 = s2 * anaJrond(y1, y2, _min = s3, _max = s2)
#         res = term1 + term2 + term3
#     return res


# def anaJrond_tmp_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = 1):
#     """
#     First intermediate step between anaJrond and (simplified) anaJrond_terms
#     """
#     def cmpi(i):
#         ti = ti_fun(i, y1, y2)
#         ri0 = rij_fun(i, 0, y1, y2, z1min, z1max, z2min, z2max)
#         ri1 = rij_fun(i, 1, y1, y2, z1min, z1max, z2min, z2max)
#         ri2 = rij_fun(i, 2, y1, y2, z1min, z1max, z2min, z2max)
#         ri3 = rij_fun(i, 3, y1, y2, z1min, z1max, z2min, z2max)
#         block1 = -s0 * s1 * np.log((ri1 + ti) / x0) - s0 * ti * np.log((ri1 + s1) / x0) + s0**2 * np.log((ri0 + ti) / x0) + ti * s0 * np.log((ri0 + s0) / x0) +  y * s0 * np.arctan2(ti * s1, y * ri1) - y * s0 * np.arctan2(ti * s0, y * ri0)
        
#         block2 = s1 * s3 * np.log((ri3 + ti) / x0) + s1 * ti * np.log((ri3 + s3) / x0) - s1**2 * np.log((ri1 + ti) / x0) - ti * s1 * np.log((ri1 + s1) / x0) -  y * s1 * np.arctan2(ti * s3, y * ri3) + y * s1 * np.arctan2(ti * s1, y * ri1)
        
#         block3 = -s0 * s3 * np.log((ri3 + ti) / x0) - s0 * ti * np.log((ri3 + s3) / x0) + s0 * s1 * np.log((ri1 + ti) / x0) + ti * s0 * np.log((ri1 + s1) / x0) +  y * s0 * np.arctan2(ti * s3, y * ri3) - y * s0 * np.arctan2(ti * s1, y * ri1)
        
#         block4 = s2**2 * np.log((ri2 + ti) / x0) + s2 * ti * np.log((ri2 + s2) / x0) - s2 * s3 * np.log((ri3 + ti) / x0) - ti * s2 * np.log((ri3 + s3) / x0) -  y * s2 * np.arctan2(ti * s2, y * ri2) + y * s2 * np.arctan2(ti * s3, y * ri3)
        
#         sm = block1 + block2 + block3 + block4
#         return (-1)**i * sm
        
#     s0 = s0_fun(z1min, z1max, z2min, z2max)
#     s1 = s1_fun(z1min, z1max, z2min, z2max)
#     s2 = s2_fun(z1min, z1max, z2min, z2max)
#     s3 = s3_fun(z1min, z1max, z2min, z2max)
#     y = y1 - y2
#     res = cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)
#     return res


# def anaJrond_tmp2_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = 1):
#     """
#     Second intermediate step between anaJrond and (simplified) anaJrond_terms
#     """
#     def cmpi(i):
#         ti = ti_fun(i, y1, y2)
#         ri0 = rij_fun(i, 0, y1, y2, z1min, z1max, z2min, z2max)
#         ri1 = rij_fun(i, 1, y1, y2, z1min, z1max, z2min, z2max)
#         ri2 = rij_fun(i, 2, y1, y2, z1min, z1max, z2min, z2max)
#         ri3 = rij_fun(i, 3, y1, y2, z1min, z1max, z2min, z2max)
#         res = s0**2 * np.log((ri0 + ti) / x0) + s0 * ti * np.log((ri0 + s0) / x0) - y * s0 * np.arctan2(ti * s0, y * ri0) + s1 * s3 * np.log((ri3 + ti) / x0) + s1 * ti * np.log((ri3 + s3) / x0) - s1**2 * np.log((ri1 + ti) / x0) - s1 * ti * np.log((ri1 + s1) / x0) - y * s1 * np.arctan2(ti * s3, y * ri3) + y * s1 * np.arctan2(ti * s1, y * ri1) - s0 * s3 * np.log((ri3 + ti) / x0) - s0 * ti * np.log((ri3 + s3) / x0) + y * s0 * np.arctan2(ti * s3, y * ri3) + s2**2 * np.log((ri2 + ti) / x0) + s2 * ti * np.log((ri2 + s2) / x0) - s2 * s3 * np.log((ri3 + ti) / x0) - s2 * ti * np.log((ri3 + s3) / x0) - y * s2 * np.arctan2(ti * s2, y * ri2) + y * s2 * np.arctan2(ti * s3, y * ri3)
#         return (-1)**i * res
        
#     s0 = s0_fun(z1min, z1max, z2min, z2max)
#     s1 = s1_fun(z1min, z1max, z2min, z2max)
#     s2 = s2_fun(z1min, z1max, z2min, z2max)
#     s3 = s3_fun(z1min, z1max, z2min, z2max)
#     y = y1 - y2
#     res = cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)
#     return res


# def anaJrond_tmp3_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = 1):
#     """
#     Second intermediate step between anaJrond and (simplified) anaJrond_terms
#     """
#     def cmpi(i):
#         ti = ti_fun(i, y1, y2)
#         ri0 = rij_fun(i, 0, y1, y2, z1min, z1max, z2min, z2max)
#         ri1 = rij_fun(i, 1, y1, y2, z1min, z1max, z2min, z2max)
#         ri2 = rij_fun(i, 2, y1, y2, z1min, z1max, z2min, z2max)
#         ri3 = rij_fun(i, 3, y1, y2, z1min, z1max, z2min, z2max)
#         f1 = s0**2 * np.log((ri0 + ti) / x0) - s1**2 * np.log((ri1 + ti) / x0) - s0 * s3 * np.log((ri3 + ti) / x0) + s2**2 * np.log((ri2 + ti) / x0) - s2 * s3 * np.log((ri3 + ti) / x0) + s1 * s3 * np.log((ri3 + ti) / x0)
#         f2 = ti * ( s0 * np.log((ri0 + s0) / x0) + s1 * np.log((ri3 + s3) / x0) - s1 * np.log((ri1 + s1) / x0) - s0 * np.log((ri3 + s3) / x0) + s2 * np.log((ri2 + s2) / x0) - s2 * np.log((ri3 + s3) / x0) )
#         f3 = y * ( -s0 * np.arctan2(ti * s0, y * ri0)  - s1 * np.arctan2(ti * s3, y * ri3) + s1 * np.arctan2(ti * s1, y * ri1)  + s0 * np.arctan2(ti * s3, y * ri3) - s2 * np.arctan2(ti * s2, y * ri2) + s2 * np.arctan2(ti * s3, y * ri3) )
#         return (-1)**i * (f1 + f2 + f3)
        
#     s0 = s0_fun(z1min, z1max, z2min, z2max)
#     s1 = s1_fun(z1min, z1max, z2min, z2max)
#     s2 = s2_fun(z1min, z1max, z2min, z2max)
#     s3 = s3_fun(z1min, z1max, z2min, z2max)
#     y = y1 - y2
#     res = cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)
#     return res


# def anaJrond_tmp4_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = 1):
#     """
#     Third intermediate step between anaJrond and (simplified) anaJrond_terms
#     """
#     def cmpi(i):
#         ti = ti_fun(i, y1, y2)
#         ri0 = rij_fun(i, 0, y1, y2, z1min, z1max, z2min, z2max)
#         ri1 = rij_fun(i, 1, y1, y2, z1min, z1max, z2min, z2max)
#         ri2 = rij_fun(i, 2, y1, y2, z1min, z1max, z2min, z2max)
#         ri3 = rij_fun(i, 3, y1, y2, z1min, z1max, z2min, z2max)
#         f1 = s0**2 * np.log((ri0 + ti) / x0) - s1**2 * np.log((ri1 + ti) / x0) + s2**2 * np.log((ri2 + ti) / x0) - s3**2 * np.log((ri3 + ti) / x0)
#         f2 = ti * (s0 * np.log((ri0 + s0) / x0) - s1 * np.log((ri1 + s1) / x0) + s2 * np.log((ri2 + s2) / x0) - s3 * np.log((ri3 + s3) / x0))
#         f3 = -y * ( s0 * np.arctan2(ti * s0, y * ri0) - s1 * np.arctan2(ti * s1, y * ri1) + s2 * np.arctan2(ti * s2, y * ri2) - s3 * np.arctan2(ti * s3, y * ri3))
#         return (-1)**i * (f1 + f2 + f3)
        
#     s0 = s0_fun(z1min, z1max, z2min, z2max)
#     s1 = s1_fun(z1min, z1max, z2min, z2max)
#     s2 = s2_fun(z1min, z1max, z2min, z2max)
#     s3 = s3_fun(z1min, z1max, z2min, z2max)
#     y = y1 - y2
#     res = cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)

#     print('---> s3, s0 - s1 + s2', s3, s0 - s1 + s2)
#     return res
    


# def numIxz_sum(y1, y2, z1min, z1max, z2min, z2max):
#     numIterms = numIrond_terms(y1, y2, z1min, z1max, z2min, z2max)
#     numJterms = numJrond_terms(y1, y2, z1min, z1max, z2min, z2max)
#     return numIterms + numJterms

# def anaIxz_sum(y1, y2, z1min, z1max, z2min, z2max, x0 = 1):
#     anaIterms = anaIrond_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
#     anaJterms = anaJrond_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
#     return anaIterms + anaJterms


def anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = 1, version = 1):
    def cmpij(i, j):
        rij = rij_fun(i, j, y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
        ti = ti_fun(i, y1, y2, a1, a2, t)
        sj = si_fun(j, z1min, z1max, z2min, z2max)
        if version == 1:
            f1 = ti * rij
            f2 = (y**2 - sj**2) * np.log((rij + ti) / x0)
            f3 = -2 * ti * sj * np.log((rij + sj) / x0)
            f4 = 2 * y * sj * np.arctan(ti * sj / (y * rij))
        elif version == 2:
            f1 = ti * rij
            f2 = np.sign(ti) * (y**2 - sj**2) * np.log((rij + np.abs(ti)) / np.sqrt(y**2 + sj**2))
            f3 = -2 * ti * sj * np.log((rij + sj) / x0)
            f4 = 2 * y * sj * np.arctan(ti * sj / (y * rij))
        else:
            raise ValueError()
        return (-1)**(i+j+1) * (f1 + f2 + f3 + f4)
        
    y = y1 - y2
    res = 0
    for i in range(4):
        for j in range(4):
            res += cmpij(i, j)
    return res / 2



def cmpIxzHoyle(t, y1, y2, s, a1, a2, h1, h2):
    """
    Eot-Wash, Hoyle et al (1999)'s Eq. B5. Modified to allow for s < 0
    """
    y = y1 - y2
    t0 = t - np.sqrt(a2**2 - y2**2) - np.sqrt(a1**2 - y1**2)
    t1 = t - np.sqrt(a2**2 - y2**2) + np.sqrt(a1**2 - y1**2)
    t2 = t + np.sqrt(a2**2 - y2**2) + np.sqrt(a1**2 - y1**2)
    t3 = t + np.sqrt(a2**2 - y2**2) - np.sqrt(a1**2 - y1**2)
    s0 = s
    s1 = s + h1
    s2 = s + h1 + h2
    s3 = s + h2
    ti = [t0, t1, t2, t3]
    sj = [s0, s1, s2, s3]
    Ixz = 0
    for i in range(4):
        for j in range(4):
            rij = np.sqrt(ti[i]**2 + y**2 + sj[j]**2)
            mone = (-1)**(i+j)
            f1 = ti[i] * rij
            f2  = np.sign(ti[i]) * (y**2 - sj[j]**2) * np.log(np.abs(ti[i]) + rij)
            if sj[j] != 0:
                f3 = -2 * ti[i] * sj[j] * np.log(np.sign(sj[j]) + rij / np.abs(sj[j]))
            else:
                f3 = 0
            f4 = 2 * y * sj[j] * np.arctan(ti[i] * sj[j] / (rij * y))
            inc = mone * (f1 + f2 + f3 + f4)
            Ixz += inc
    Ixz *= -0.5
    return Ixz



########################
####### Fz #############
########################
## integral over z1 and z2 (Iz)
def numIz_z2(y1_in, y2_in, x, z1min, z1max, z2min, z2max):
    """
    2D numerical integration of V(x, y, z1, z2).dz1.dz2
    """
    #dv = lambda x1, x2: dVx(x1, x2, y, z)
    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)
    if n > 1:
        res = np.zeros(n)
        for i in range(n):
            #resi, err = dblquad(dVz, z1min, z1max, z2min, z2max, args = (y[i], x))
            resi, err = dblquad(dVz, z2min, z2max, z1min, z1max, args = (y[i], x)) #think bounds are correct like that
            res[i] = resi
    else:
        #res, err = dblquad(dVz, z1min, z1max, z2min, z2max, args = (y, x))
        res, err = dblquad(dVz, z2min, z2max, z1min, z1max, args = (y, x)) #think bounds are correct like that
    return res


def numIz(y1_in, y2_in, x, z1min, z1max, z2min, z2max, yukawa = False, lmbda = None):
    """
    1D numerical integration of V(x1, x2, y, z).dx1.dx2
    """
    #vf = lambda x: V(x, y1 - y2, z)
    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)

    s0 = s0_fun(z1min, z1max, z2min, z2max)
    s1 = s1_fun(z1min, z1max, z2min, z2max)
    s2 = s2_fun(z1min, z1max, z2min, z2max)
    s3 = s3_fun(z1min, z1max, z2min, z2max)
    if not yukawa:
        if n > 1:
            res1 = np.zeros(n)
            res2 = np.zeros(n)
            for i in range(n):
                res1[i], err1 = quad(Vz, s3, s2, args = (y[i], x))
                res2[i], err2 = quad(Vz, s0, s1, args = (y[i], x))
        else:
            res1, err1 = quad(Vz, s3, s2, args = (y, x))
            res2, err2 = quad(Vz, s0, s1, args = (y, x))
    else:
        if lmbda is None:
            raise TypeError("lmbda must be set to work with Yukawa")
        if n > 1:
            res1 = np.zeros(n)
            res2 = np.zeros(n)
            for i in range(n):
                res1[i], err1 = quad(VzYuk, s3, s2, args = (y[i], x, lmbda))
                res2[i], err2 = quad(VzYuk, s0, s1, args = (y[i], x, lmbda))
        else:
            res1, err1 = quad(VzYuk, s3, s2, args = (y, x, lmbda))
            res2, err2 = quad(VzYuk, s0, s1, args = (y, x, lmbda))
    return res1 - res2


def anaIz(y1, y2, x, z1min, z1max, z2min, z2max, x0 = 1, version = 2):
    """
    Analytic integration for Ix
    x0 is artificial to have dimensionless quantities inside logs and should not affect the result
    """
    def cmpi(i):
        si = si_fun(i, z1min, z1max, z2min, z2max)
        if version == 1:
            return (-1)**i * np.log((np.sqrt(x**2 + y**2 + si**2) + si) / x0)
        elif version == 2:
            #if y**2 + z**2 > 0:
            num = np.sqrt(x**2 + y**2 + si**2) + np.abs(si)
            denom = np.sqrt(x**2 + y**2)
            #else:
            #    return (-1)**i * np.log((np.sqrt(ti**2 + y**2 + z**2) + ti) / x0)
            return (-1)**i * np.sign(si) * np.log(num / denom)
        else:
            raise ValueError()

    y = y1 - y2
    res = cmpi(0) + cmpi(1) + cmpi(2) + cmpi(3)
    return res


## integral over z (Izx)
def numIz_z2_x1x2(x1, x2, y1, y2, z1min, z1max, z2min, z2max):
    return numIz_z2(y1, y2, x1 - x2, z1min, z1max, z2min, z2max)

def numIzx_x2(y1_in, y2_in, a1, a2, t, z1min, z1max, z2min, z2max):
    """
    2D numerical integration of Iz(y1, y2, x1, x2).dx1.dx2
    To be consistent, use 2D numerical integration of V(x,y,z) to get Iz
    """
    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)
    #Ix = numIx_x2(y1, y2, z, a1, a2, t)

    x1min = x1min_fun(a1, y1)
    x1max = x1max_fun(a1, y1)
    #if n > 1:
    #    res = np.zeros(n)
    #    for i in range(n):
    #        resi, err = dblquad(dVx, x1min[i], x1max[i], x2min_fun(a2, y2[i], t), x2max_fun(a2, y2[i], t), args = (y[i], z))
    
    if n > 1:
        res = np.zeros(n)
        for i in range(n):
            resi, err = dblquad(numIz_z2_x1x2, x1min[i], x1max[i], x2min_fun(a2, y2[i], t), x2max_fun(a2, y2[i], t), args = (y1[i], y2[i], z1min, z1max, z2min, z2max))
            
            res[i] = resi
    else:
        res, err = dblquad(numIz_z2_x1x2, x1min, x1max, x2min_fun(a2, y2, t), x2max_fun(a2, y2, t), args = (y1, y2, z1min, z1max, z2min, z2max))
        #res, err = dblquad(numIx_x2_z1z2, z1min, z1max, z2min, z2max, args = (y1, y2, a1, a2, t))
    return res


def numIzx(y1_in, y2_in, a1, a2, t, z1min, z1max, z2min, z2max, yukawa = False, lmbda = None):
    """
    2D numerical integration of Iz(y1, y2, x1, x2).dx1.dx2
    To be consistent, use 2D numerical integration of V(x,y,z) to get Iz
    """
    if np.size(y1_in) != np.size(y2_in):
        if np.size(y1_in) > 1:
            if np.size(y2_in) == 1:
                y1 = np.copy(y1_in)
                y2 = np.tile(y2_in, np.size(y1))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        elif np.size(y2_in) > 1:
            if np.size(y1_in) == 1:
                y2 = np.copy(y2_in)
                y1 = np.tile(y1_in, np.size(y2))
            else:
                raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
        else:
            raise TypeError('y1 and y2 must have same size, or one of them must be a scalar')
    else:
        y1 = y1_in
        y2 = y2_in
        
    y = y1 - y2
    n = np.size(y)
    t0 = t0_fun(y1, y2, a1, a2, t)
    t1 = t1_fun(y1, y2, a1, a2, t)
    t2 = t2_fun(y1, y2, a1, a2, t)
    t3 = t3_fun(y1, y2, a1, a2, t)
    #Ix = numIx_x2(y1, y2, z, a1, a2, t)
    if n > 1:
        res = np.zeros(n)
        for i in range(n):
            i_term1 = numIrond_z(y1[i], y2[i], z1min, z1max, z2min, z2max, _min = t0[i], _max = t1[i], yukawa = yukawa, lmbda = lmbda)
            i_term2 = -numIrond_z(y1[i], y2[i], z1min, z1max, z2min, z2max, _min = t3[i], _max = t2[i], yukawa = yukawa, lmbda = lmbda)
            j_term1 = -t0[i] * numJrond_z(y1[i], y2[i], z1min, z1max, z2min, z2max, _min = t0[i], _max = t1[i], yukawa = yukawa, lmbda = lmbda)
            j_term2 = (t1[i] - t0[i]) * numJrond_z(y1[i], y2[i], z1min, z1max, z2min, z2max, _min = t1[i], _max = t3[i], yukawa = yukawa, lmbda = lmbda)
            j_term3 = t2[i] * numJrond_z(y1[i], y2[i], z1min, z1max, z2min, z2max, _min = t3[i], _max = t2[i], yukawa = yukawa, lmbda = lmbda)
            res[i] = i_term1 + i_term2 + j_term1 + j_term2 +j_term3
    else:
        i_term1 = numIrond_z(y1, y2, z1min, z1max, z2min, z2max, _min = t0, _max = t1, yukawa = yukawa, lmbda = lmbda)
        i_term2 = -numIrond_z(y1, y2, z1min, z1max, z2min, z2max, _min = t3, _max = t2, yukawa = yukawa, lmbda = lmbda)
        j_term1 = -t0 * numJrond_z(y1, y2, z1min, z1max, z2min, z2max, _min = t0, _max = t1, yukawa = yukawa, lmbda = lmbda)
        j_term2 = (t1 - t0) * numJrond_z(y1, y2, z1min, z1max, z2min, z2max, _min = t1, _max = t3, yukawa = yukawa, lmbda = lmbda)
        j_term3 = t2 * numJrond_z(y1, y2, z1min, z1max, z2min, z2max, _min = t3, _max = t2, yukawa = yukawa, lmbda = lmbda)
        res = i_term1 + i_term2 + j_term1 + j_term2 +j_term3
    
    return res

#def numIx_z(z, y1, y2, a1, a2, t):
#    return numIx(y1, y2, z, a1, a2, t)

def numIrond_z(y1, y2, z1min, z1max, z2min, z2max, _min = None, _max = None, yukawa = False, lmbda = None):
    """
    1D numerical integration of x*Iz(x).dx that enters in Izx
    For consistency, use 1D numerical integration of Iz
    """
    int_x = lambda x: x * numIz(y1, y2, x, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
    if _min is None:
        raise NotImplementedError()
    if _max is None:
        raise NotImplementedError()
    res, err = quad(int_x, _min, _max)
    return res

def numJrond_z(y1, y2, z1min, z1max, z2min, z2max, _min = None, _max = None, yukawa = False, lmbda = None):
    """
    Numerical integration of Ix(z).dz that enters in Ixz
    For consistency, use 1D numerical integration of Ix
    """
    int_x = lambda x: numIz(y1, y2, x, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
    if _min is None:
        raise NotImplementedError()
    if _max is None:
        raise NotImplementedError()
    res, err = quad(int_x, _min, _max)
    return res

def anaIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = 1, version = 1):
    def cmpij(i, j):
        rij = rij_fun(j, i, y1, y2, a1, a2, t, z1min, z1max, z2min, z2max) #invert i and j to have tj and si in rij instead of ti and sj
        tj = ti_fun(j, y1, y2, a1, a2, t)
        si = si_fun(i, z1min, z1max, z2min, z2max)
        if version == 1:
            f1 = si * rij
            f2 = (y**2 - tj**2) * np.log((rij + si) / x0)
            f3 = -2 * si * tj * np.log((rij + tj) / x0)
            f4 = 2 * y * tj * np.arctan(si * tj / (y * rij))
        elif version == 2:
            f1 = si * rij
            f2 = np.sign(si) * (y**2 - tj**2) * np.log((rij + np.abs(si)) / np.sqrt(y**2 + tj**2))
            f3 = -2 * si * tj * np.log((rij + tj) / x0)
            f4 = 2 * y * tj * np.arctan(si * tj / (y * rij))
        else:
            raise ValueError()
        return (-1)**(i+j+1) * (f1 + f2 + f3 + f4)
        
    y = y1 - y2
    res = 0
    for i in range(4):
        for j in range(4):
            res += cmpij(i, j)
    return res / 2


###########################
### potential integrals ###
###########################
def integrateV(y, z, xmin = 1, xmax = 2, verbose = True):
    """
    Compare numeric and analytic integration of V=1/sqrt(x^2+y^2+z^2)
    """
    int_x = lambda x: 1./np.sqrt(x**2 + y**2 + z**2)
    
    if y**2 + z**2 == 0:
        print('hello...')
        ana = np.sign(xmax) * np.log(np.abs(xmax)) - np.sign(xmin) * np.log(np.abs(xmin))
        #anaAbs = np.sign(xmax) * np.log(np.abs(np.sqrt(xmax**2 + y**2 + z**2) + xmax)) - np.sign(xmin) * np.log(np.abs(np.sqrt(xmin**2 + y**2 + z**2) + xmin))
    else:
        if np.sqrt(xmax**2 + y**2 + z**2) + xmax <= 0:
            print("Baaad (xmax)!!!!")
        if np.sqrt(xmin**2 + y**2 + z**2) + xmin <= 0:
            print("Baaad (xmin)!!!!")
        xvec = np.linspace(xmin, xmax, 5000)
        inlog = np.sqrt(xvec**2 + y**2 + z**2) + xvec
        bd = np.where(inlog <= 0)[0]
        if np.size(bd) > 0:
            print("Some xs have np.sqrt(x**2 + y**2 + z**2) + x <= 0 !")

        #ana = np.sign(xmax) * np.log(np.sqrt(xmax**2 + y**2 + z**2) + xmax) - np.sign(xmin) * np.log(np.sqrt(xmin**2 + y**2 + z**2) + xmin)
        ana = np.log(np.sqrt(xmax**2 + y**2 + z**2) + xmax) - np.log(np.sqrt(xmin**2 + y**2 + z**2) + xmin)
        #anaAbs = np.log(np.sqrt(xmax**2 + y**2 + z**2) + xmax) - np.log(np.sqrt(xmin**2 + y**2 + z**2) + xmin)

    num, err = quad(int_x, xmin, xmax)

    #if verbose:
    #    print('integrate V')
    #    print('   bounds (log):', np.sqrt(xmax**2 + y**2 + z**2) + xmax, np.sqrt(xmin**2 + y**2 + z**2) + xmin)
    #    print('   bounds (log --abs):', np.sqrt(xmax**2 + y**2 + z**2) + np.abs(xmax), np.sqrt(xmin**2 + y**2 + z**2) + np.abs(xmin))
    #    print('   ana, anaAbs:', ana, anaAbs)
    #    print('   num:', num)

    #if y**2 + z**2 == 0:
    #    return ana, num
    #else:
    #    return anaAbs, num
    return ana, num


def pltIntegrateV(ny = 400, nz = 400, xmin = 1, xmax = 2, log = False, ylim = [-1, 1], zlim = [-1, 1]):
    #first, plot including (y,z)=(0,0)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[1], nz)
    intg_ana = np.zeros((ny, nz))
    intg_num = np.zeros((ny, nz))
    for i in range(ny):
        for j in range(nz):
            ana, num = integrateV(y[i], z[j], xmin = xmin, xmax = xmax, verbose = False)
            intg_ana[i,j] = ana
            intg_num[i,j] = num
    intg_ana = np.flipud(intg_ana)
    intg_num = np.flipud(intg_num)
    if log:
        intg_ana = np.log(np.abs(intg_ana))
        intg_num = np.log(np.abs(intg_num))
    plt.subplot(221)
    plt.imshow(intg_ana, extent = [z[0], z[-1], y[0], y[-1]])
    plt.xlabel('z')
    plt.ylabel('y')
    cbar = plt.colorbar()
    if not log:
        cbar.set_label('ana')
    else:
        cbar.set_label('log(ana)')
    plt.subplot(222)
    plt.imshow(intg_num, extent = [z[0], z[-1], y[0], y[-1]])
    plt.xlabel('z')
    plt.ylabel('y')
    cbar = plt.colorbar()
    if not log:
        cbar.set_label('num')
    else:
        cbar.set_label('log(num)')

    #then, plot without (y,z)=(0,0)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[0] + 0.25 * (zlim[1]-zlim[0]), nz)
    intg_ana = np.zeros((ny, nz))
    intg_num = np.zeros((ny, nz))
    for i in range(ny):
        for j in range(nz):
            ana, num = integrateV(y[i], z[j], xmin = xmin, xmax = xmax, verbose = False)
            intg_ana[i,j] = ana
            intg_num[i,j] = num
    intg_ana = np.flipud(intg_ana)
    intg_num = np.flipud(intg_num)
    if log:
        intg_ana = np.log(np.abs(intg_ana))
        intg_num = np.log(np.abs(intg_num))
    plt.subplot(223)
    plt.imshow(intg_ana, extent = [z[0], z[-1], y[0], y[-1]])
    plt.xlabel('z')
    plt.ylabel('y')
    cbar = plt.colorbar()
    if not log:
        cbar.set_label('ana')
    else:
        cbar.set_label('log(ana)')
    plt.subplot(224)
    plt.imshow(intg_num, extent = [z[0], z[-1], y[0], y[-1]])
    plt.xlabel('z')
    plt.ylabel('y')
    cbar = plt.colorbar()
    if not log:
        cbar.set_label('num')
    else:
        cbar.set_label('log(num)')
    plt.suptitle('xmin, xmax = ' + str(xmin) + ' ' + str(xmax))
    plt.show(block = True)
    
    

### tests for Fx ###

def checkIx(theta1 = 0.1, theta2 = 0.3, z = 0, a1 = 1.1, a2 = 2.3, t = 0.1, x0 = 1):
    y1 = a1 * np.sin(theta1) #this makes sure that y1 and y2 are realistic
    y2 = a2 * np.sin(theta2)
    num2 = numIx_x2(y1, y2, z, a1, a2, t)
    num = numIx(y1, y2, z, a1, a2, t)
    ana1 = anaIx(y1, y2, z, a1, a2, t, x0 = x0, version = 1)
    ana2 = anaIx(y1, y2, z, a1, a2, t, x0 = x0, version = 2)
    print('Ix:', num2, num, ana1, ana2)



def checkIxz(theta1 = 0.1, theta2 = 0.3, z1min = -1, z1max = 1, z2min = -1.3, z2max = 1.3, a1 = 1.1, a2 = 2.3, t = 0.1, x0 = 1):
    y1 = a1 * np.sin(theta1) #this makes sure that y1 and y2 are realistic
    y2 = a2 * np.sin(theta2)
    num2 = numIxz_z2(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
    num = numIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
    ana1 = anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
    ana2 = anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
    print('Ixz:', num2, num, ana1, ana2)
    

def pltIx(n, t, y2 = None, a1 = 1.1, a2 = 2.3, z = 0, x0 = 1):
    pltIx_xz('Ix', n, t, y2 = y2, a1 = a1, a2 = a2, z = z, x0 = x0)

def pltIxz(n, t, y2 = None, a1 = 1.1, a2 = 2.3, z1min = -1, z1max = 1, z2min = -1.3, z2max = 1.3, x0 = 1, inner_a1 = 0, inner_a2 = 0):
    pltIx_xz('Ixz', n, t, y2 = y2, a1 = a1, a2 = a2, z1min = z1min, z1max = z1max, z2min = z2min, z2max = z2max, x0 = x0, inner_a1 = inner_a1, inner_a2 = inner_a2)
    
def pltIx_xz(_kind, n, t, y2 = None, a1 = 1.1, a2 = 2.3, z = 0, z1min = -1, z1max = 1, z2min = -1.3, z2max = 1.3, x0 = 1, inner_a1 = 0, inner_a2 = 0):
    if inner_a1 > 0 or inner_a2 > 0:
        raise NotImplementedError('I think it is incorrect, and perhaps not worth pursuing...')
    theta = np.linspace(-np.pi/2, np.pi/2, n)
    y1 = a1 * np.sin(theta)
    if inner_a1 > 0:
        y1 = inner_a1 * np.sin(theta)
    if y2 is None:
        y2 = a2 * np.sin(theta)
        if inner_a2 > 0:
            y2 = inner_a2 * np.sin(theta)
        _type = '2D'
    else:
        if not (isinstance(y2, float) or isinstance(y2, int)):
            raise TypeError('y2 must be a float (if not None)')
        _type = '1D'

        
    #first, compute things
    if _type == '1D':
        if _kind == 'Ix':
            num2 = numIx_x2(y1, y2, z, a1, a2, t)
            num = numIx(y1, y2, z, a1, a2, t)
            ana1 = anaIx(y1, y2, z, a1, a2, t, x0 = x0, version = 1)
            ana2 = anaIx(y1, y2, z, a1, a2, t, x0 = x0, version = 2)
        elif _kind == 'Ixz':
            num2 = numIxz_z2(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
            num = numIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
            ana1 = anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
            ana2 = anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
            if inner_a1 > 0 or inner_a2 > 0:
                num2_inner = numIxz_z2(y1, y2, inner_a1, inner_a2, t, z1min, z1max, z2min, z2max)
                num_inner = numIxz(y1, y2, inner_a1, inner_a2, t, z1min, z1max, z2min, z2max)
                ana1_inner = anaIxz(y1, y2, inner_a1, inner_a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
                ana2_inner = anaIxz(y1, y2, inner_a1, inner_a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
                num2 -= num2_inner
                num -= num_inner
                ana1 -= ana1_inner
                ana2 -= ana2_inner
        else:
            raise ValueError()

    elif _type == '2D':
        num2 = np.zeros((n,n))
        num = np.zeros((n,n))
        ana1 = np.zeros((n,n))
        ana2 = np.zeros((n,n))
        if inner_a1 > 0 or inner_a2 > 0:
            num2_inner = np.zeros((n,n))
            num_inner = np.zeros((n,n))
            ana1_inner = np.zeros((n,n))
            ana2_inner = np.zeros((n,n))
        if _kind == 'Ix':
            for i in range(n):
                num2[:,i] = numIx_x2(y1, y2[i], z, a1, a2, t)
                num[:,i] = numIx(y1, y2[i], z, a1, a2, t)
                ana1[:,i] = anaIx(y1, y2[i], z, a1, a2, t, x0 = x0, version = 1)
                ana2[:,i] = anaIx(y1, y2[i], z, a1, a2, t, x0 = x0, version = 2)
        elif _kind == 'Ixz':
            for i in range(n):
                num2[:,i] = numIxz_z2(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max)
                num[:,i] = numIxz(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max)
                ana1[:,i] = anaIxz(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
                ana2[:,i] = anaIxz(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
                if inner_a1 > 0 or inner_a2 > 0:
                    num2_inner[:,i] = numIxz_z2(y1, y2[i], inner_a1, inner_a2, t, z1min, z1max, z2min, z2max)
                    num_inner[:,i] = numIxz(y1, y2[i], inner_a1, inner_a2, t, z1min, z1max, z2min, z2max)
                    ana1_inner[:,i] = anaIxz(y1, y2[i], inner_a1, inner_a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
                    ana2_inner[:,i] = anaIxz(y1, y2[i], inner_a1, inner_a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
            if inner_a1 > 0 or inner_a2 > 0:
                num2 -= num2_inner
                num -= num_inner
                ana1 -= ana1_inner
                ana2 -= ana2_inner
        else:
            raise ValueError()
        
    else:
        raise ValueError('Bad!!!')

    #now, plot
    if _type == '1D':
        ax = plt.subplot(211)
        plt.plot(y1, num2, label = '2D num')
        plt.plot(y1, num, linestyle = '--', label = '1D num')
        plt.plot(y1, ana1, linestyle = '-.', label = 'ana #1')
        plt.plot(y1, ana2, linestyle = ':', label = 'ana #2')
        plt.ylabel(_kind)
        plt.legend()
        plt.subplot(212, sharex = ax)
        plt.plot(y1, num - num2, label = '1D num')
        plt.plot(y1, ana1 - num2, linestyle = '--', label = 'ana #1')
        plt.plot(y1, ana2 - num2, linestyle = '-.', label = 'ana #2')
        plt.ylabel(_kind + ' - 2Dnum')
        plt.legend()
        if _kind == 'Ix':
            plt.suptitle(_kind + ', (y2, t, z)=(' + str(y2) + ',' + str(t) + ',' + str(z) + ')')
        else:
            plt.suptitle(_kind + ', (y2, t, zbounds)=(' + str(y2) + ',' + str(t) + ', [' + str(z1min) + ',' + str(z1max) + ',' + str(z2min) + ',' + str(z2max) + '])')
        plt.show(block = True)
    else:
        num2 = np.flipud(num2)
        num = np.flipud(num)
        ana1 = np.flipud(ana1)
        ana2 = np.flipud(ana2)
        ax = plt.subplot(241)
        plt.imshow(num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', 2D num')
        plt.subplot(242, sharex = ax, sharey = ax)
        plt.imshow(num, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', 1D num')
        plt.subplot(243, sharex = ax, sharey = ax)
        plt.imshow(ana1, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', ana #1')
        plt.subplot(244, sharex = ax, sharey = ax)
        plt.imshow(ana2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', ana #2')
        plt.subplot(246, sharex = ax, sharey = ax)
        plt.imshow(num - num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + '-2Dnum, 1D num')
        plt.subplot(247, sharex = ax, sharey = ax)
        plt.imshow(ana1 - num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + '-2Dnum, ana #1')
        plt.subplot(248, sharex = ax, sharey = ax)
        plt.imshow(ana2 - num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + '-2Dnum, ana #2')
        if _kind == 'Ix':
            plt.suptitle(_kind + ', (t, z)=(' + str(t) + ',' + str(z) + ')')
        else:
            plt.suptitle(_kind + ', (t, zbounds)=(' + str(t) + ', [' + str(z1min) + ',' + str(z1max) + ',' + str(z2min) + ',' + str(z2max) + '])')
        #plt.suptitle(_kind + ', (t, z)=(' + str(t) + ',' + str(z) + ')')
        plt.show(block = True)



### tests for Fz ###
def checkIz(theta1 = 0.1, theta2 = 0.3, x = 0, z1min = -1, z1max = 1, z2min = -1.3, z2max = 1.3, a1 = 1.1, a2 = 2.3, x0 = 1):
    y1 = a1 * np.sin(theta1) #this makes sure that y1 and y2 are realistic
    y2 = a2 * np.sin(theta2)
    num2 = numIz_z2(y1, y2, x, z1min, z1max, z2min, z2max)
    num = numIz(y1, y2, x, z1min, z1max, z2min, z2max)
    ana1 = anaIz(y1, y2, x, z1min, z1max, z2min, z2max, x0 = 1, version = 1)
    ana2 = anaIz(y1, y2, x, z1min, z1max, z2min, z2max, x0 = 1, version = 2)
    print('Iz:', num2, num, ana1, ana2)

def checkIzx(theta1 = 0.1, theta2 = 0.3, z1min = -1, z1max = 1, z2min = -1.3, z2max = 1.3, a1 = 1.1, a2 = 2.3, t = 0.1, x0 = 1):
    y1 = a1 * np.sin(theta1) #this makes sure that y1 and y2 are realistic
    y2 = a2 * np.sin(theta2)
    num2 = numIzx_x2(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
    num = numIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
    ana1 = anaIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
    ana2 = anaIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
    print('Izx:', num2, num, ana1, ana2)
    
def pltIz(n, dz, y2 = None, t = 0, a1 = 1.1, a2 = 2.3, h1 = 3, h2 = 4, x0 = 1):
    pltIz_zx('Iz', n, dz, y2 = y2, t = t, a1 = a1, a2 = a2, h1 = h1, h2 = h2, x0 = x0)
        
def pltIzx(n, dz, y2 = None, t = 0, a1 = 1.1, a2 = 2.3, h1 = 2, h2 = 4, x0 = 1, inner_a1 = 0, inner_a2 = 0):
    pltIz_zx('Izx', n, dz, y2 = y2, t = t, a1 = a1, a2 = a2, h1 = h1, h2 = h2, x0 = x0)
    
def pltIz_zx(_kind, n, dz, y2 = None, t = 0, dx = 0, a1 = 1.1, a2 = 2.3, h1 = 3, h2 = 4, x0 = 1):
    theta = np.linspace(-np.pi/2, np.pi/2, n)
    y1 = a1 * np.sin(theta)
    if y2 is None:
        y2 = a2 * np.sin(theta)
        _type = '2D'
    else:
        if not (isinstance(y2, float) or isinstance(y2, int)):
            raise TypeError('y2 must be a float (if not None)')
        _type = '1D'

    # define z-bounds
    z1 = 0
    z1min = z1 - 0.5 * h1
    z1max = z1 + 0.5 * h1
    z2 = z1 + dz
    z2min = z2 - 0.5 * h2
    z2max = z2 + 0.5 * h2
        
    #first, compute things
    x = t + dx
    if _type == '1D':
        if _kind == 'Iz':
            z1 = 0.5 * (z1min + z1max)
            z2 = 0.5 * (z2min + z2max)
            dz = z1 - z2
            num2 = numIz_z2(y1, y2, x, z1min, z1max, z2min, z2max)
            num = numIz(y1, y2, x, z1min, z1max, z2min, z2max)
            ana1 = anaIz(y1, y2, x, z1min, z1max, z2min, z2max, x0 = 1, version = 1)
            ana2 = anaIz(y1, y2, x, z1min, z1max, z2min, z2max, x0 = 1, version = 2)
        elif _kind == 'Izx':
            num2 = numIzx_x2(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
            num = numIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
            ana1 = anaIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
            ana2 = anaIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
        else:
            raise ValueError()

    elif _type == '2D':
        num2 = np.zeros((n,n))
        num = np.zeros((n,n))
        ana1 = np.zeros((n,n))
        ana2 = np.zeros((n,n))
        if _kind == 'Iz':
            for i in range(n):
                num2[:,i]  = numIz_z2(y1, y2[i], x, z1min, z1max, z2min, z2max)
                num[:,i]  = numIz(y1, y2[i], x, z1min, z1max, z2min, z2max)
                ana1[:,i]  = anaIz(y1, y2[i], x, z1min, z1max, z2min, z2max, x0 = 1, version = 1)
                ana2[:,i]  = anaIz(y1, y2[i], x, z1min, z1max, z2min, z2max, x0 = 1, version = 2)
        elif _kind == 'Izx':
            for i in range(n):
                num2[:,i] = numIzx_x2(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max)
                num[:,i] = numIzx(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max)
                ana1[:,i] = anaIzx(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 1)
                ana2[:,i] = anaIzx(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
        else:
            raise ValueError()
        
    else:
        raise ValueError('Bad!!!')

    #now, plot
    if _type == '1D':
        ax = plt.subplot(211)
        plt.plot(y1, num2, label = '2D num')
        plt.plot(y1, num, linestyle = '--', label = '1D num')
        plt.plot(y1, ana1, linestyle = '-.', label = 'ana #1')
        plt.plot(y1, ana2, linestyle = ':', label = 'ana #2')
        plt.ylabel(_kind)
        plt.legend()
        plt.subplot(212, sharex = ax)
        plt.plot(y1, num - num2, label = '1D num')
        plt.plot(y1, ana1 - num2, linestyle = '--', label = 'ana #1')
        plt.plot(y1, ana2 - num2, linestyle = '-.', label = 'ana #2')
        plt.ylabel(_kind + ' - 2Dnum')
        plt.legend()
        plt.xlabel('y1')
        if _kind == 'Iz':
            plt.suptitle(_kind + ', (y2, x, dz)=(' + str(y2) + ',' + str(x) + ',' + str(dz) + ')')
        else:
            plt.suptitle(_kind + ', (y2, t, zbounds)=(' + str(y2) + ',' + str(t) + ', [' + str(z1min) + ',' + str(z1max) + ',' + str(z2min) + ',' + str(z2max) + '])')
        plt.show(block = True)
    else:
        num2 = np.flipud(num2)
        num = np.flipud(num)
        ana1 = np.flipud(ana1)
        ana2 = np.flipud(ana2)
        ax = plt.subplot(241)
        plt.imshow(num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', 2D num')
        plt.subplot(242, sharex = ax, sharey = ax)
        plt.imshow(num, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', 1D num')
        plt.subplot(243, sharex = ax, sharey = ax)
        plt.imshow(ana1, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', ana #1')
        plt.subplot(244, sharex = ax, sharey = ax)
        plt.imshow(ana2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + ', ana #2')
        plt.subplot(246, sharex = ax, sharey = ax)
        plt.imshow(num - num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + '-2Dnum, 1D num')
        plt.subplot(247, sharex = ax, sharey = ax)
        plt.imshow(ana1 - num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + '-2Dnum, ana #1')
        plt.subplot(248, sharex = ax, sharey = ax)
        plt.imshow(ana2 - num2, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        cbar = plt.colorbar()
        cbar.set_label(_kind + '-2Dnum, ana #2')
        if _kind == 'Iz':
            plt.suptitle(_kind + ', (x, dz)=(' + str(x) + ',' + str(dz) + ')')
        else:
            plt.suptitle(_kind + ', (t, zbounds)=(' + str(t) + ', [' + str(z1min) + ',' + str(z1max) + ',' + str(z2min) + ',' + str(z2max) + '])')
        #plt.suptitle(_kind + ', (t, z)=(' + str(t) + ',' + str(z) + ')')
        plt.show(block = True)
    

    
    
def tstIx(theta1 = 0.1, theta2 = 0.3, z = 0, x0 = 1, z1min = -1, z1max = -0.7, z2min = 0.3, z2max = 0.5):
    y1 = a1 * np.sin(theta1) #this makes sure that y1 and y2 are realistic
    y2 = a2 * np.sin(theta2)
    print(y1,y2)
    s0 = s0_fun(z1min, z1max, z2min, z2max)
    s1 = s1_fun(z1min, z1max, z2min, z2max)
    s2 = s2_fun(z1min, z1max, z2min, z2max)
    s3 = s3_fun(z1min, z1max, z2min, z2max)
    print('---> s3, s0 - s1 + s2', s3, s0 - s1 + s2)

    num2 = numIx_x2(y1, y2, z)
    num = numIx(y1, y2, z)
    ana = anaIx(y1, y2, z, x0 = x0)
    print('Ix:', num2, num, ana)

    numIr = numIrond(y1, y2, _min = -1, _max = 1.1)
    anaIr = anaIrond(y1, y2, _min = -1, _max = 1.1, x0 = x0)
    print('Irond:', numIr, anaIr)

    numJr = numJrond(y1, y2, _min = -1, _max = 1.1)
    anaJr = anaJrond(y1, y2, _min = -1, _max = 1.1, x0 = x0)
    print('Jrond:', numJr, anaJr)

    numIterms = numIrond_terms(y1, y2, z1min, z1max, z2min, z2max)
    anaIterms = anaIrond_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    anaIterms_v2 = anaIrond_terms_v1(y1, y2, z1min, z1max, z2min, z2max, x0 = x0, v2 = True)
    print('Ixz Irond terms:', numIterms, anaIterms, anaIterms_v2)

    numJterms = numJrond_terms(y1, y2, z1min, z1max, z2min, z2max)
    anaJterms = anaJrond_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    anaJterms_tmp = anaJrond_tmp_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    anaJterms_tmp2 = anaJrond_tmp2_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    anaJterms_tmp3 = anaJrond_tmp3_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    anaJterms_tmp4 = anaJrond_tmp4_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    anaJterms_v2 = anaJrond_terms(y1, y2, z1min, z1max, z2min, z2max, x0 = x0, v2 = True)
    print('Ixz Jrond terms:', numJterms, anaJterms, anaJterms_tmp, anaJterms_tmp2, anaJterms_tmp3, anaJterms_tmp4, anaJterms_v2)

    numS = numIxz_sum(y1, y2, z1min, z1max, z2min, z2max)
    anaS = anaIxz_sum(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    ana = anaIxz(y1, y2, z1min, z1max, z2min, z2max, x0 = x0)
    print('Ixz:', numS, anaS, ana)

    anal = analIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0)
    print('Anal Ixz:', anal)

    #compare with Hoyle
    deltaZ = 0.5 * (z2min + z2max) - 0.5 * (z1min + z1max) #distance between cylinders centers (2-1)
    h1 = z1max - z1min
    h2 = z2max - z2min
    s = deltaZ - 0.5 * (h1 + h2)
    #s = -s0
    #h1 = -(s1 - s0)
    #h2 = -(s3 - s0)
    hoyle = cmpIxzHoyle(t, y1, y2, s, a1, a2, h1, h2)
    print('Hoyle:', hoyle)

    print()
    print('z1 bounds:', z1min, z1max, z1max - z1min)
    print('z2 bounds:', z2min, z2max, z2max - z2min)
    print('cyl. centers:', 0.5 * (z1min + z1max), 0.5 * (z2min + z2max))
    print('distance between cylinders centers (2-1): ', deltaZ)
    print('distance between top of 1 and bottom of 2:', s)
    print('s, h1, h2:', s, h1, h2)

    
    s = 0
    for i in range(4):
        for j in range(4):
            ti = ti_fun(i, y1, y2)
            sj = si_fun(j, z1min, z1max, z2min, z2max)
            s += (-1)**(i+j) * ti * sj
    print(s)
