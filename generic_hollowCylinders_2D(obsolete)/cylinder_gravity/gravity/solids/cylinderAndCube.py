"""
Compute force between a hollow cylinder and a cube
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import interp1d
from island.gravity import cylinder, parallelepipoid

class cycu:
    def __init__(self, R, e, h, Lx, Ly, Lz, rho_cyl = 1, rho_cube = 1):
        self.cyl = cylinder.cylinder(R+e, h, density = rho_cyl, originType = 'centered', innerRadius = R)
        self.cube = parallelepipoid.parallelepipoid(Lx, Ly, Lz, density = rho_cube, originType = 'centered')

    def a_grid(self, x = 0, y = 0, z = 0, source = 'cylinder', alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        compute acceleration created by cylinder or cube on a 1D, 2D or 3D grid
        source -- 'cylinder' or 'cube'
        """
        if source == 'cylinder':
            x, y, z, ax, ay, az = self.cyl.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        elif source == 'cube':
            x, y, z, ax, ay, az = self.cube.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        else:
            raise ValueError('Bad cylid!')
        return x, y, z, ax, ay, az


    def cmpCubeForce(self, d, ignoreEdgeEffects = True, nr = 30, nz1 = 20, nx = 100, ny = 100, nz = 100, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Compute (radial) force created by cylinder on cube as the cube moves along x
        """
        if np.size(d) == 1:
            if not (np.abs(d) < self.cyl.innerRadius - 0.5 * self.cube.length or np.abs(d) >  self.cyl.radius + 0.5 * self.cube.length):
                raise ValueError("Displacement d=" + str(d) + "m not allowed by geometry.")
            self.cube.setOrigin([d, 0., 0.])
            d = [d]
        else:
            bd = np.where((np.abs(d) < self.cyl.innerRadius - 0.5 * self.cube.length) & (np.abs(d) >  self.cyl.radius + 0.5 * self.cube.length))[0]
            if np.size(bd) > 0:
                raise ValueError("Displacement not allowed by geometry.")

        #compute radial acceleration created by silica cylinders between them
        #if np.abs(d) <= self.cyl.innerRadius - 0.5 * self.cube.length:
        r = np.linspace(0, 10 * self.cyl.radius, nr)
        #else:
        #    r = np.linspace(self.cyl.radius, 10 * self.cyl.radius, nr)
        x1 = r
        y1 = 0
        if not ignoreEdgeEffects:
            zmin = 1.1 * min(-0.5*self.cyl.height, -0.5*self.cyl.height)
            zmax = 1.1 * max(0.5*self.cyl.height, 0.5*self.cyl.height)
            z = np.linspace(zmin, zmax, nz1)
            raise NotImplementedError()
        else:
            z = 0
        x1, y1, z1, ax1, ay1, az1 = self.a_grid(x1, y1, z, source = 'cylinder', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)

        #grid of points inside the cube, together with their mass (set to 0 outside the test mass, obviously)
        Fx = np.zeros(np.size(d))
        Fy = np.zeros(np.size(d))
        for i in range(np.size(d)):
            self.cube.setOrigin([d[i], 0., 0.])
            x = np.linspace(-0.6 * self.cube.length, 0.6 * self.cube.length, nx) + self.cube.x0
            y = np.linspace(-0.6 * self.cube.width, 0.6 * self.cube.width, ny) + self.cube.y0
            z = np.linspace(-0.5*self.cube.height, 0.5*self.cube.height, nz) + self.cube .z0
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            dz = z[1] - z[0]
            cellV = dx * dy * dz
            cellM = self.cube.density * cellV
 
            #... first, compute force at z=0 (assuming vertical translation invariance)
            xp, yp = np.meshgrid(x, y)
            rp = np.sqrt(xp**2 + yp**2)
            thetap = np.arctan2(yp, xp)
            funx = interp1d(r, ax1, kind = 'linear', bounds_error = False, fill_value = 0)
            funy = interp1d(r, ay1, kind = 'linear', bounds_error = False, fill_value = 0)
            #ax = np.cos(thetap) * funx(rp) - np.sin(thetap) * funy(rp)
            #ay = np.sin(thetap) * funx(rp) + np.cos(thetap) * funy(rp)
            ax = np.cos(thetap) * funx(rp) #funx(rp) is ax taken along the x-axis, i.e. the (signed) norm of the acceleration; this is exactly like the full rotation above, but with the symmetry assumption y->-y explicit
            ay = np.sin(thetap) * funx(rp)

            #compute force in z=0 plane
            xl = x - 0.5 * dx
            xr = x + 0.5 * dx
            yl = y - 0.5 * dy
            yr = y + 0.5 * dy
            zl = - 0.5 * dz
            zr = 0.5 * dz
            yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
            yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
        
            in_it = self.cube.inside((xl3, yl3, zl3), (xr3, yr3, zr3))
            Fx[i] = cellM * np.sum(np.transpose(ax) * in_it[:,:,0])
            Fy[i] = cellM * np.sum(np.transpose(ay) * in_it[:,:,0])
        if np.size(d) == 1:
            Fx = Fx[0]
            Fy = Fy[0]

        #... and sum over height
        Fx *= nz
        Fy *= nz

        #... and remember that we assume that the test mass is perfectly centered about z=0 and that it is parallel to silica cylinders
        if np.size(d) == 1:
            Fz = 0
        else:
            Fz = np.zeros(np.size(d))

        return Fx, Fy, Fz


    def cmpCylinderForce(self, d, ignoreEdgeEffects = True, nr = 30, nz1 = 20, nx = 100, ny = 100, nz = 100, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Compute (radial) force created by cube on cylinder as the cylinder moves along x
        """
        if np.size(d) == 1:
            if not (np.abs(d) < self.cyl.innerRadius - 0.5 * self.cube.length or np.abs(d) >  self.cyl.radius + 0.5 * self.cube.length):
                raise ValueError("Displacement d=" + str(d) + "m not allowed by geometry.")
            self.cyl.setOrigin([d, 0., 0.])
            d = [d]
        else:
            bd = np.where((np.abs(d) < self.cyl.innerRadius - 0.5 * self.cube.length) & (np.abs(d) >  self.cyl.radius + 0.5 * self.cube.length))[0]
            if np.size(bd) > 0:
                raise ValueError("Displacement not allowed by geometry.")

        #compute radial acceleration created by silica cylinders between them
        #if np.abs(d) <= self.cyl.innerRadius - 0.5 * self.cube.length:
        r = np.linspace(0, 100 * self.cube.length, nr)
        #else:
        #    r = np.linspace(self.cyl.radius, 10 * self.cyl.radius, nr)
        x1 = r
        y1 = 0
        if not ignoreEdgeEffects:
            zmin = 1.1 * min(-0.5*self.cyl.height, -0.5*self.cyl.height)
            zmax = 1.1 * max(0.5*self.cyl.height, 0.5*self.cyl.height)
            z = np.linspace(zmin, zmax, nz1)
            raise NotImplementedError()
        else:
            z = 0
        x1, y1, z1, ax1, ay1, az1 = self.a_grid(x1, y1, z, source = 'cube', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)

        #grid of points inside the cube, together with their mass (set to 0 outside the test mass, obviously)
        Fx = np.zeros(np.size(d))
        Fy = np.zeros(np.size(d))
        for i in range(np.size(d)):
            self.cyl.setOrigin([d[i], 0., 0.])
            x = np.linspace(-1.1 * self.cyl.radius, 1.1 * self.cyl.radius, nx) + self.cyl.x0
            y = np.linspace(-1.1 * self.cyl.radius,1.1 * self.cyl.radius, ny) + self.cyl.y0
            z = np.linspace(-0.5*self.cyl.height, 0.5*self.cyl.height, nz) + self.cyl.z0
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            dz = z[1] - z[0]
            cellV = dx * dy * dz
            cellM = self.cyl.density * cellV
 
            #... first, compute force at z=0 (assuming vertical translation invariance)
            xp, yp = np.meshgrid(x, y)
            rp = np.sqrt(xp**2 + yp**2)
            thetap = np.arctan2(yp, xp)
            funx = interp1d(r, ax1, kind = 'linear', bounds_error = False, fill_value = 0)
            funy = interp1d(r, ay1, kind = 'linear', bounds_error = False, fill_value = 0)
            #ax = np.cos(thetap) * funx(rp) - np.sin(thetap) * funy(rp)
            #ay = np.sin(thetap) * funx(rp) + np.cos(thetap) * funy(rp)
            ax = np.cos(thetap) * funx(rp) #funx(rp) is ax taken along the x-axis, i.e. the (signed) norm of the acceleration; this is exactly like the full rotation above, but with the symmetry assumption y->-y explicit
            ay = np.sin(thetap) * funx(rp)

            #compute force in z=0 plane
            xl = x - 0.5 * dx
            xr = x + 0.5 * dx
            yl = y - 0.5 * dy
            yr = y + 0.5 * dy
            zl = - 0.5 * dz
            zr = 0.5 * dz
            yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
            yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
        
            in_it = self.cyl.inside((xl3, yl3, zl3), (xr3, yr3, zr3))
            Fx[i] = cellM * np.sum(np.transpose(ax) * in_it[:,:,0])
            Fy[i] = cellM * np.sum(np.transpose(ay) * in_it[:,:,0])
        if np.size(d) == 1:
            Fx = Fx[0]
            Fy = Fy[0]

        #... and sum over height
        Fx *= nz
        Fy *= nz

        #... and remember that we assume that the test mass is perfectly centered about z=0 and that it is parallel to silica cylinders
        if np.size(d) == 1:
            Fz = 0
        else:
            Fz = np.zeros(np.size(d))

        return Fx, Fy, Fz



def cmpForce_d(source, inout = 'in', nr = 30, nz1 = 20, nx = 100, ny = 100, nz = 100, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01):
    R = 10e-2
    e = 5e-3
    h = 15e-2
    Lx = 1e-2
    Ly = 1e-2
    Lz = 1e-2
    if inout == 'in':
        d = np.linspace(-0.95*(R+e)+0.5*Lx, 0.95*(R+e)-0.5*Lx, 50)
    elif inout == 'out':
        dp = np.linspace(1.05*(R+e)+0.5*Lx, 5*(R+e), 50)
        dm = np.linspace(-5*(R+e), -1.05*(R+e)+0.5*Lx, 50)
        d = np.append(dm, dp)
    else:
        raise ValueError('Bad inout')
    c = cycu(R, e, h, Lx, Ly, Lz, rho_cyl = 2000, rho_cube = 2000)
    if source == 'cylinder':
        #force created by cylinder on cube
        Fx, Fy, Fz = c.cmpCubeForce(d, nx = nx, ny = ny, nz = nz, nr = nr, nz1 = nz1, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    elif source == 'cube':
        #force created by cube on cylinder
        Fx, Fy, Fz = c.cmpCylinderForce(d, nx = nx, ny = ny, nz = nz, nr = nr, nz1 = nz1, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    else:
        raise ValueError('Bad source!')

    ax = plt.subplot(211)
    plt.plot(d, Fx)
    plt.ylabel('Fx [N]')
    plt.subplot(212, sharex = ax)
    plt.plot(d, Fy)
    plt.ylabel('Fy [N]')
    plt.xlabel('d [m]')
    plt.show(block = True)


