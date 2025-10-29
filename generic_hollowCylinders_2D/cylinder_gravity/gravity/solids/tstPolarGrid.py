import numpy as np

def pmesh2D(ntheta = 8, nr = 7):
    theta = np.linspace(0, 2 * np.pi, ntheta)
    r = np.linspace(0, 2, nr)
    th2, r2 = np.meshgrid(theta, r)
    print(theta)
    print(r)
    print(th2)
    print(r2)
    
