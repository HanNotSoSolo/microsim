import numpy as np
from scipy.integrate import quad, dblquad

def DV(x1, x2):
    y1 = 0
    y2 = 0
    z1 = 0
    z2 = 1
    return (x2 - x1) / ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** (3./2)

def V(x):
    y1 = 0
    y2 = 0
    z1 = 0
    z2 = 1
    return 1. / np.sqrt(x**2 + (y2 - y1)**2 + (z2 - z1)**2)

def tst(t = 3.5, a1 = 1, a2 = 0.5, y1 = 0.3, y2 = -0.1):
    f2min = lambda x: t - np.sqrt(a2**2 - y2**2)
    f2max = lambda x: t + np.sqrt(a2**2 - y2**2)
    x1min = -np.sqrt(a1**2 - y1**2)
    x1max = np.sqrt(a1**2 - y1**2)
    x2min = t - np.sqrt(a2**2 - y2**2)
    x2max = t + np.sqrt(a2**2 - y2**2)
    int2d, err2d = dblquad(DV, x1min, x1max, f2min, f2max)
    int1d2, err1d2 = quad(V, x2min, x2max)
    int1d1, err1d1 = quad(V, x1min, x1max)
    int1d = int1d2 - int1d1

    #Eot-Wash
    t0 = t - np.sqrt(a2**2 - y2**2) - np.sqrt(a1**2 - y1**2)
    t1 = t - np.sqrt(a2**2 - y2**2) + np.sqrt(a1**2 - y1**2)
    t2 = t + np.sqrt(a2**2 - y2**2) + np.sqrt(a1**2 - y1**2)
    t3 = t + np.sqrt(a2**2 - y2**2) - np.sqrt(a1**2 - y1**2)
    intew1, errew1 = quad(V, t3, t2)
    intew2, errew2 = quad(V, t0, t1)
    intew = intew1 - intew2

    print(int2d, int1d, intew)
