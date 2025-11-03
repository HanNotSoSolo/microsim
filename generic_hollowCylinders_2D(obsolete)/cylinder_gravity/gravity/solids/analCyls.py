import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def x1min_fun(a1, y1):
    """
    Compute lower integration boundary on x for cylinder #1
    cylinder is assumed centered on (0,0,0)
    a1 -- cylinder's radius
    y1 -- y-coordinate of volume elements
    """
    return -np.sqrt(a1**2 - y1**2)

def x1max_fun(a1, y1):
    """
    Compute upper integration boundary on x for cylinder #1
    cylinder is assumed centered on (0,0,0)
    a1 -- cylinder's radius
    y1 -- y-coordinate of volume elements
    """
    return np.sqrt(a1**2 - y1**2)

def x2min_fun(a2, y2, t):
    """
    Compute lower integration boundary on x for cylinder #2
    t -- x-coordinate of cylinder
    a2 -- cylinder's radius
    y2 -- y-coordinate of volume elements
    """
    return t - np.sqrt(a2**2 - y2**2)

def x2max_fun(a2, y2, t):
    """
    Compute upper integration boundary on x for cylinder #2
    t -- x-coordinate of cylinder
    a2 -- cylinder's radius
    y2 -- y-coordinate of volume elements
    """
    return t + np.sqrt(a2**2 - y2**2)

def t0_fun(y1, y2, a1, a2, t):
    return x2min_fun(a2, y2, t) - x1max_fun(a1, y1)

def t1_fun(y1, y2, a1, a2, t):
    return x2min_fun(a2, y2, t) - x1min_fun(a1, y1)

def t2_fun(y1, y2, a1, a2, t):
    return x2max_fun(a2, y2, t) - x1min_fun(a1, y1)

def t3_fun(y1, y2, a1, a2, t):
    return x2max_fun(a2, y2, t) - x1max_fun(a1, y1)

def ti_fun(i, y1, y2, a1, a2, t):
    if i == 0:
        return t0_fun(y1, y2, a1, a2, t)
    elif i == 1:
        return t1_fun(y1, y2, a1, a2, t)
    elif i == 2:
        return t2_fun(y1, y2, a1, a2, t)
    elif i == 3:
        return t3_fun(y1, y2, a1, a2, t)
    else:
        raise ValueError('Bad i')

def s0_fun(z1min, z1max, z2min, z2max):
    """
    z?min, z?max -- integration boundaries for cylinders
    """
    return z1min - z2max

def s1_fun(z1min, z1max, z2min, z2max):
    """
    z?min, z?max -- integration boundaries for cylinders
    """
    return z1min - z2min

def s2_fun(z1min, z1max, z2min, z2max):
    """
    z?min, z?max -- integration boundaries for cylinders
    """
    return z1max - z2min

def s3_fun(z1min, z1max, z2min, z2max):
    """
    z?min, z?max -- integration boundaries for cylinders
    """
    return z1max - z2max

def si_fun(i, z1min, z1max, z2min, z2max):
    if i == 0:
        return s0_fun(z1min, z1max, z2min, z2max)
    elif i == 1:
        return s1_fun(z1min, z1max, z2min, z2max)
    elif i == 2:
        return s2_fun(z1min, z1max, z2min, z2max)
    elif i == 3:
        return s3_fun(z1min, z1max, z2min, z2max)
    else:
        raise ValueError('Bad i')

def rij_fun(i, j, y1, y2, a1, a2, t, z1min, z1max, z2min, z2max):
    y = y1 - y2
    ti = ti_fun(i, y1, y2, a1, a2, t)
    sj = si_fun(j, z1min, z1max, z2min, z2max)
    rij = np.sqrt(ti**2 + y**2 + sj**2)
    return rij

def anaIx(y1, y2, a1, a2, t, z, x0 = 1):
    """
    Analytic integration for Ix
    x0 is artificial to have dimensionless quantities inside logs and should not affect the result
    """
    y = y1 - y2
    bad = np.where(y**2 + z**2 == 0)[0]
    if np.size(bad) > 0:
        print('WARNING! analCyls.anaIx: some y^2 + z^2 = 0')
    t0 = ti_fun(0, y1, y2, a1, a2, t)
    t1 = ti_fun(1, y1, y2, a1, a2, t)
    t2 = ti_fun(2, y1, y2, a1, a2, t)
    t3 = ti_fun(3, y1, y2, a1, a2, t)
    f0 = np.log((np.sqrt(t0**2 + y**2 + z**2) + t0) / x0)
    f1 = np.log((np.sqrt(t1**2 + y**2 + z**2) + t1) / x0)
    f2 = np.log((np.sqrt(t2**2 + y**2 + z**2) + t2) / x0)
    f3 = np.log((np.sqrt(t3**2 + y**2 + z**2) + t3) / x0)
    return f0 - f1 + f2 - f3

def anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = 1, eps = 1e-1):
    def cmpij(i, j):
        rij = rij_fun(i, j, y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
        ti = ti_fun(i, y1, y2, a1, a2, t)
        sj = si_fun(j, z1min, z1max, z2min, z2max)
        #f1 = f2 = f3 = f4 = 0
        f1 = ti * rij
        #f2 = np.sign(ti) * (y**2 - sj**2) * np.log((rij + np.abs(ti)) / x0)
        f2 = np.sign(t) * (y**2 - sj**2) * np.log((rij + ti) / x0)
        #f2 = (y**2 - sj**2) * np.log((rij + np.abs(ti)) / x0)
        #f2 = np.sign(ti) * (y**2 - sj**2) * np.log((rij + ti) / x0)
        f3 = -2 * ti * sj * np.log((rij + sj) / x0)
        f4 = 2 * y * sj * np.arctan(ti * sj / (y * rij))
        #print("---> anaIxz (loop)")
        #print(y1, y2, a1, a2, t, i, j, np.sign(ti), ti, (y**2 - sj**2) * np.log((rij + np.abs(ti)) / x0), f2)
        #print("  ", rij, ti, rij + np.abs(ti), np.log(rij + np.abs(ti)))
        #print("  ", x2max_fun(a2, y2, t), x1max_fun(a1, y1), x2max_fun(a2, y2, t) - x1max_fun(a1, y1))
        #print((rij + np.abs(ti)) / x0, (rij + sj) / x0)
        #print(f1, f2, f3, f4)
        return (-1)**(i+j+1) * (f1 + f2 + f3 + f4)

    y = y1 - y2
    res = 0
    for i in range(4):
        for j in range(4):
            res += cmpij(i, j)
    #print("---> anaIxz", res/2)
    #print(y1, y2, a1, a2, x2max_fun(a2, y2, t), x1max_fun(a1, y1), x2max_fun(a2, y2, t) - x1max_fun(a1, y1), t3_fun(y1, y2, a1, a2, t))
    #print()
    rij = rij_fun(3, 3, y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
    ti = t3_fun(y1, y2, a1, a2, t)
    return res / 2#, np.log(rij + np.abs(ti))#, t3_fun(y1, y2, a1, a2, t)


def numIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max):
    numIterms = numIrond_terms(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
    numJterms = numJrond_terms(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max)
    return numIterms + numJterms


def numIrond(y1, y2, a1, a2, t, _min = None, _max = None):
    """
    Numerical integration of z*Ix(z).dz that enters in Ixz
    """
    int_z = lambda z: z * anaIx(y1, y2, a1, a2, t, z)
    if _min is None:
        raise NotImplementedError()
    if _max is None:
        raise NotImplementedError()
    res, err = quad(int_z, _min, _max)
    return res

def numJrond(y1, y2, a1, a2, t, _min = None, _max = None):
    """
    Numerical integration of Ix(z).dz that enters in Ixz
    """
    int_z = lambda z: anaIx(y1, y2, a1, a2, t, z)
    if _min is None:
        raise NotImplementedError()
    if _max is None:
        raise NotImplementedError()
    res, err = quad(int_z, _min, _max)
    return res

def numIrond_terms(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max):
    s0 = si_fun(0, z1min, z1max, z2min, z2max)
    s1 = si_fun(1, z1min, z1max, z2min, z2max)
    s2 = si_fun(2, z1min, z1max, z2min, z2max)
    s3 = si_fun(3, z1min, z1max, z2min, z2max)
    term1 = numIrond(y1, y2, a1, a2, t, _min = s0, _max = s1)
    term2 = numIrond(y1, y2, a1, a2, t, _min = s3, _max = s2)
    return term1 - term2

def numJrond_terms(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max):
    s0 = si_fun(0, z1min, z1max, z2min, z2max)
    s1 = si_fun(1, z1min, z1max, z2min, z2max)
    s2 = si_fun(2, z1min, z1max, z2min, z2max)
    s3 = si_fun(3, z1min, z1max, z2min, z2max)
    term1 = -s0 * numJrond(y1, y2, a1, a2, t, _min = s0, _max = s1)
    term2 = (s1 - s0) * numJrond(y1, y2, a1, a2, t, _min = s1, _max = s3)
    term3 = s2 * numJrond(y1, y2, a1, a2, t, _min = s3, _max = s2)
    return term1 + term2 + term3

def pltIx(n, t, y2 = None, a1 = 1.1, a2 = 2.3, z = 0, x0 = 1):
    theta = np.linspace(-np.pi/2, np.pi/2, n)
    y1 = a1 * np.sin(theta)
    if y2 is None:
        y2 = a2 * np.sin(theta)
        _type = '2D'
    else:
        if not (isinstance(y2, float) or isinstance(y2, int)):
            raise TypeError('y2 must be a float (if not None)')
        _type = '1D'

    if _type == '1D':
        Ix = anaIx(y1, y2, a1, a2, t, z, x0 = x0)
        plt.plot(y1, Ix)
        plt.title('Ix, (y2, t, z)=(' + str(y2) + ',' + str(t) + ',' + str(z) + ')')
        plt.show(block = True)
    elif _type == '2D':
        Ix = np.zeros((n,n))
        for i in range(n):
            Ix[:,i] = anaIx(y1, y2[i], a1, a2, t, z, x0 = x0)
        Ix = np.flipud(Ix)
        plt.imshow(Ix, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        plt.colorbar()
        plt.suptitle('Ix, (t, z)=(' + str(t) + ',' + str(z) + ')')
        plt.show(block = True)
    else:
        raise ValueError('Bad!!!')
    

def pltIxz(n, t, y2 = None, a1 = 1.1, a2 = 2.3, z1min = -0.8, z1max = 0.8, z2min = -0.7, z2max = 0.7, x0 = 1, numeric = False):
    theta = np.linspace(-np.pi/2, np.pi/2, n)
    y1 = a1 * np.sin(theta)
    if y2 is None:
        y2 = a2 * np.sin(theta)
        _type = '2D'
    else:
        if not (isinstance(y2, float) or isinstance(y2, int)):
            raise TypeError('y2 must be a float (if not None)')
        _type = '1D'

    dz = 0.5 * ((z2min + z2max) - (z1min + z1max))
    if _type == '1D':
        if not numeric:
            Ixz = anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0)
        else:
            Ixz = np.zeros(n)
            for i in range(n):
                Ixz[i] = numIxz(y1[i], y2, a1, a2, t, z1min, z1max, z2min, z2max)
                print(Ixz[i])
        plt.plot(y1, Ixz)
        plt.title('Ixz, (y2, t, dz)=(' + str(y2) + ',' + str(t) + ',' + str(dz) + ')')
        plt.show(block = True)
    elif _type == '2D':
        Ixz = np.zeros((n,n))
        for i in range(n):
            if not numeric:
                Ixz[:,i] = anaIxz(y1, y2[i], a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0)
            else:
                for j in range(n):
                    Ixz[j,i] = numIxz(y1[j], y2[i], a1, a2, t, z1min, z1max, z2min, z2max)
            plt.plot(y1, Ixz[:,i])
        plt.xlabel('y1')
        plt.ylabel('Ixz')
        plt.title('Ixz, (t, dz)=(' + str(t) + ',' + str(dz) + ')')
        plt.show(block = True)
        
        Ixz = np.flipud(Ixz)
        plt.imshow(Ixz, extent = [y1[0], y1[-1], y2[0], y2[-1]])
        plt.xlabel('y1')
        plt.ylabel('y2')
        plt.colorbar()
        plt.suptitle('Ixz, (t, dz)=(' + str(t) + ',' + str(dz) + ')')
        plt.show(block = True)
    else:
        raise ValueError('Bad!!!')
    

def plt_tis(n, t, y2 = None, a1 = 1.1, a2 = 2.3):
    theta = np.linspace(-np.pi/2, np.pi/2, n)
    y1 = a1 * np.sin(theta)
    if y2 is None:
        y2 = a2 * np.sin(theta)
        _type = '2D'
    else:
        if not (isinstance(y2, float) or isinstance(y2, int)):
            raise TypeError('y2 must be a float (if not None)')
        _type = '1D'

    ax = plt.subplot(221)
    if _type == '1D':
        for i in range(4):
            if i > 0:
                plt.subplot(2,2,i, sharex = ax)
            ti = ti_fun(i, y1, y2, a1, a2, t)
            plt.plot(y1, ti)
            plt.ylabel('t' + str(i))
        plt.title('ti, (y2, t)=(' + str(y2) + ',' + str(t) + ')')
        plt.show(block = True)
        
    elif _type == '2D':
        for i in range(4):
            ti = np.zeros((n,n))
            if i > 0:
                plt.subplot(2,2,i+1, sharex = ax)
            for j in range(n):
                ti[:,j] = ti_fun(i, y1, y2[j], a1, a2, t)
            ti = np.flipud(ti)
            plt.imshow(ti, extent = [y1[0], y1[-1], y2[0], y2[-1]])
            plt.xlabel('y1')
            plt.ylabel('y2')
            cbar = plt.colorbar()
            cbar.set_label('t'+str(i))
        plt.suptitle('ti, t=' + str(t))
        plt.show(block = True)
    else:
        raise ValueError('Bad!!!')
    

