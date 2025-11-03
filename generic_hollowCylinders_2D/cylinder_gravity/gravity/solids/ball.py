import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.integrate import simpson
#try:
#    from skmonaco import mcquad #scikit-monaco
#except:
#    print("WARNING! scikit-monaco not found, and seems obsolete. Cannot perform Monte Carlo integration (or need a new package)")
from scipy.constants import G
from scipy.integrate import tplquad
from scipy.interpolate import interp1d
import time
from gravity.solids import solid

class ball(solid.solid):
    def __init__(self, radius, origin = [0.,0.,0.], density = 1, innerRadius = 0):
        """
        innerRadius -- if >0, defines a hollow ball (shell)
        """
        super(ball, self).__init__(density, origin = origin)
        self.kind = 'ball'
        self.radius = radius
        self.innerRadius = innerRadius
        self.hollow = (self.innerRadius > 0)
        self.tplquad_xmin = self.x0 - self.radius #lower boundary for x for tplquad integration
        self.tplquad_xmax = self.x0 + self.radius #upper boundary for x for tplquad integration

    def tplquad_ymin_fun(self, x):
        """lower boundary for y for tplquad integration (gfun in terms of tplquad doc)"""
        tx = x - self.x0
        return self.y0 - np.sqrt(self.radius**2 - tx**2)

    def tplquad_ymax_fun(self, x):
        """upper boundary for y for tplquad integration (hfun in terms of tplquad doc)"""
        tx = x - self.x0
        return self.y0 + np.sqrt(self.radius**2 - tx**2)

    def tplquad_zmin_fun(self, x, y):
        """lower boundary for z for tplquad integration (qfun in terms of tplquad doc)"""
        print('... not implemented...')
        raise NotImplementedError()

    def tplquad_zmax_fun(self, x, y):
        """upper boundary for z for tplquad integration (rfun in terms of tplquad doc)"""
        print('... not implemented...')
        raise NotImplementedError()

    def cmpVolume(self):
        if not self.hollow:
            return 4./3 * np.pi * self.radius**3
        else:
            return 4./3 * np.pi * (self.radius**3 - self.innerRadius**3)

        
    def inside(self, xyzmin, xyzmax, strictBorder = True):
        """
        Check if a small volume (or an ensemble of small volumes) is inside or outside the cylinder. Return 1 for volume(s) inside the cylinder, 0 for others
        xyzmin -- (xleft, yleft, zmin)
        xyzmax -- (xright, yright, zmax)
        strictBorder -- if set, a small cube is said to be inside the cylinder if it is entirely inside. Otherwise, said to be inside even if it is halfway out.
        """
        xleft = xyzmin[0] - self.x0
        yleft = xyzmin[1] - self.y0
        zleft = xyzmin[2] - self.z0
        xright = xyzmax[0] - self.x0
        yright = xyzmax[1] - self.y0
        zright = xyzmax[2] - self.z0

        rleft = np.sqrt(xleft**2 + yleft**2 + zleft**2)
        rright = np.sqrt(xright**2 + yright**2 + zright**2)
        rmin = np.minimum(rleft, rright)
        rmax = np.maximum(rleft, rright)

        if np.size(xleft) == 1:
            if strictBorder:
                if rmax <= self.radius and rmin >= self.innerRadius:
                    return 1
                else:
                    return 0
            else:
                if rmin <= self.radius and rmax >= self.innerRadius:
                    return 1
                else:
                    return 0
        else:
            output = np.zeros(np.shape(xleft))
            if strictBorder:
                inner = np.where((rmax <= self.radius) & (rmin >= self.innerRadius))#[0]
            else:
                inner = np.where((rmin <= self.radius) & (rmax >= self.innerRadius))#[0]
            output[inner] = 1
            return output

    def insidePoint(self, x_in, y_in, z_in):
        """
        Check if a point (or an ensemble of points) is inside or outside the ball/shell. Return 1 for point(s) inside the cylinder, 0 for others
        """
        x = x_in - self.x0
        y = y_in - self.y0
        z = z_in - self.z0
        if np.size(x) == 1:
            if np.sqrt(x**2 + y**2 + z**2) <= self.radius and np.sqrt(x**2 + y**2 + z**2) >= self.innerRadius:
                return 1
            else:
                return 0
        else:
            r = np.sqrt(x**2 + y**2 + z**2)
            output = np.zeros(np.shape(x))
            inner = np.where((r <= self.radius) & (r >= self.innerRadius))#[0]
            output[inner] = 1
            return output

    def mkGrid(self, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, xmin = None, xmax = None, ymin = None, ymax = None, zmin = None, zmax = None, borders = True):
        """
        Make regular grid inside cylinder
        dx, dy, dz -- relative size (wrt to cylinder size) of volume elements
        """
        dx = self.radius * dx_rel
        dy = self.radius * dy_rel
        dz = self.radius * dz_rel

        if xmin is None or xmax is None:
            xspan = 2 * 1.01 * self.radius
            nx = int(xspan / dx) + 1
            x = np.linspace(-1.01 * self.radius, 1.01 * self.radius, nx) + self.x0
        else:
            nx = int((xmax - xmin) / dx + 1)
            x = np.linspace(xmin, xmax, nx)
        if ymin is None or ymax is None:
            yspan = 2 * 1.01 * self.radius
            ny = int(yspan / dy) + 1
            y = np.linspace(-1.01 * self.radius, 1.01 * self.radius, ny) + self.y0
        else:
            ny = int((ymax - ymin) / dy + 1)
            y = np.linspace(ymin, ymax, ny)
        if zmin is None or zmax is None:
            zspan = 2 * 1.01 * self.radius
            nz = int(zspan / dz) + 1
            z = np.linspace(-1.01 * self.radius, 1.01 * self.radius, nz) + self.z0
        else:
            nz = int((zmax - zmin) / dz + 1)
            z = np.linspace(zmin, zmax, nz)
        
        if borders:
            xl = x - 0.5 * dx
            xr = x + 0.5 * dx
            yl = y - 0.5 * dy
            yr = y + 0.5 * dy
            zl = z - 0.5 * dz
            zr = z + 0.5 * dz
            return x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz
        else:
            return x, nx, y, ny, z, nz


    def inertiaMoment(self, i, j, k, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, method = 'simps', mc_npoints = 1e6, memoryFriendly = False, verbose = True, fullNumeric = False):
        """
        Compute ijk inertia moment (Int[rho*x^i*y^j*z^k dxdydz])
        method -- tplquad or simps
        """
        if i >= 4 or j >= 4 or k >= 4 or fullNumeric:
            #I didn't do any analytic computation in this case, so use the numerical implementation
            return self.inertiaMoment_num(i, j, k, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, method = method, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        else:
            if i == 0 and j == 0:
                Iij = 2 * np.pi
            elif i == 0 and j == 2:
                Iij = np.pi
            elif i == 2 and j == 0:
                Iij = np.pi
            elif i == 2 and j == 2:
                Iij = np.pi / 4
            else:
                Iij = 0

            if i == 0 and j == 0 and k == 0:
                Jijk = 2
            elif i == 0 and j == 0 and k == 2:
                Jijk = 2./3
            elif i == 0 and j == 1 and k == 0:
                Jijk = np.pi / 2
            elif i == 0 and j == 1 and k == 2:
                Jijk = np.pi / 8
            elif i == 0 and j == 2 and k == 0:
                Jijk = 4./3
            elif i == 0 and j == 2 and k == 2:
                Jijk = 4./15
            elif i == 0 and j == 3 and k == 0:
                Jijk = 3*np.pi / 8
            elif i == 0 and j == 3 and k == 2:
                Jijk = np.pi / 16
            elif i == 1 and j == 0 and k == 0:
                Jijk = np.pi/2
            elif i == 1 and j == 0 and k == 2:
                Jijk = np.pi/8
            elif i == 1 and j == 1 and k == 0:
                Jijk = 4./3
            elif i == 1 and j == 1 and k == 2:
                Jijk = 4./15
            elif i == 1 and j == 2 and k == 0:
                Jijk = 3*np.pi/8
            elif i == 1 and j == 2 and k == 2:
                Jijk = np.pi/16
            elif i == 1 and j == 3 and k == 0:
                Jijk = 16./15
            elif i == 1 and j == 3 and k == 2:
                Jijk = 16./105
            elif i == 2 and j == 0 and k == 0:
                Jijk = 4./3
            elif i == 2 and j == 0 and k == 2:
                Jijk = 4./15
            elif i == 2 and j == 1 and k == 0:
                Jijk = 3*np.pi/8
            elif i == 2 and j == 1 and k == 2:
                Jijk = np.pi / 16
            elif i == 2 and j == 2 and k == 0:
                Jijk = 16./15
            elif i == 2 and j == 2 and k == 2:
                Jijk = 16./105
            elif i == 2 and j == 3 and k == 0:
                Jijk = 5*np.pi / 16
            elif i == 2 and j == 3 and k == 2:
                Jijk = 5*np.pi / 128
            elif i == 3 and j == 0 and k == 0:
                Jijk = 3*np.pi / 8
            elif i == 3 and j == 0 and k == 2:
                Jijk = np.pi / 16
            elif i == 3 and j == 1 and k == 0:
                Jijk = 16./15
            elif i == 3 and j == 1 and k == 2:
                Jijk = 16./105
            elif i == 3 and j == 2 and k == 0:
                Jijk = 5*np.pi / 16
            elif i == 3 and j == 2 and k == 2:
                Jijk = 5*np.pi / 128
            elif i == 3 and j == 3 and k == 0:
                Jijk = 32./35
            elif i == 3 and j == 3 and k == 2:
                Jijk = 32./315
            else:
                Jijk = 0
                                
            f1 = (self.radius**(i+j+k+3) - self.innerRadius**(i+j+k+3)) / (i+j+k+3)
            I = f1 * Jijk * Iij

            if i >= 10 or j >= 10 or k >= 10:
                raise NotImplementedError('i>=10, j>=10, k>10 not ready')
            else:
                iname = 'I' + str(i) + str(j) + str(k)
            setattr(self, iname, I)
            return I
        
    def cmpPotential_XYZ_mcquad(self, XYZ, npoints = 1e6, verbose = True, timeit = False):
        """
        XYZ -- (X, Y, Z)
        """
        raise NotImplementedError('Gotta find a new Monte Carlo integrator')

    # def cmpPotential_XYZ_inertiaMoments(self, XYZ, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, im_method = 'simps', memoryFriendly = False, timeit = False):
    #     """
    #     XYZ -- (X, Y, Z)
    #     memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
    #     """
    #     raise NotImplementedError()

    def cmpPotential_ana(self, XYZ):
        """
        analytical potential at X,Y,Z
        XYZ -- (X, Y, Z)
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        r = np.sqrt((X - self.x0)**2 + (Y - self.y0)**2 + (Z - self.z0)**2)
        if np.size(r) == 1:
            if r <= self.radius:
                return -G * self.cmpMass() / self.radius
            else:
                return -G * self.cmpMass() / r
        else:
            V = -G * self.cmpMass() / r
            inside = np.where(r <= self.radius)[0]
            V[inside] = -G * self.cmpMass() / self.radius
            return V

    def cmpAcceleration_ana(self, XYZ):
        """
        analytical potential at X,Y,Z
        XYZ -- (X, Y, Z)
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        r = np.sqrt((X - self.x0)**2 + (Y - self.y0)**2 + (Z - self.z0)**2)
        if np.size(r) == 1:
            if r <= self.radius:
                norm = 0
            else:
                norm = -G * self.cmpMass() / r**3
        else:
            norm = -G * self.cmpMass() / r**3
            inside = np.where(r <= self.radius)[0]
            norm[inside] = 0
        ax = norm * (X-self.x0)
        ay = norm * (Y-self.y0)
        az = norm * (Z-self.z0)
        return ax, ay, az


    def plt_vr(self, rmin, rmax, nr = 100, theta = 0, phi = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential as a function of radial distance from center of cylinder
        rmin, rmax - r limits [m]
        theta -- polar angle coordinate
        z -- altitude
        """
        if rmax <= rmin:
            raise ValueError("rmin must be < rmax")
        if not log:
            r = np.linspace(rmin, rmax, nr)
        else:
            if rmin <= 0:
                raise ValueError("For log, rmin must be strictly >0")
            r = np.logspace(np.log10(rmin), np.log10(rmax), nr)
        if np.abs(theta) < 1e-6 and np.abs(phi) < 1e-6:
            x = r - self.x0
            y = self.y0
            z = self.z0
        else:
            x = r * np.sin(theta) * np.cos(phi) - self.x0
            y = r * np.sin(theta) * np.sin(theta) - self.y0
            z = r * np.cos(theta) - self.z0

        x, y, z, v = self.v_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        v_ana = self.cmpPotential_ana((x,y,z))
        if yscale == 'log':
            v = np.abs(v)
        plt.plot(r, v, color = 'black', label = 'num')
        plt.plot(r, v_ana, color = 'blue', linestyle = '--', label = 'ana')
        ylim = plt.ylim()
        if not log and r[0] < 0:
            plt.plot([-self.radius, -self.radius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        plt.plot([self.radius, self.radius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        if self.hollow:
            if not log and r[0] < 0:
                plt.plot([-self.innerRadius, -self.innerRadius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            plt.plot([self.innerRadius, self.innerRadius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        if log:
            plt.xscale('log')
        plt.yscale(yscale)
        plt.xlabel('r (m)')
        plt.ylabel(r'$V(r, z=' + str(z) + ')$ [$J.kg^{-1}$]')
        plt.legend()
        plt.tight_layout()
        plt.show(block = True)
        

    def plt_ar(self, rmin, rmax, nr = 100, theta = 0, phi = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot acceleration as a function of radial distance from center of cylinder
        rmin, rmax - r limits [m]
        theta -- polar angle coordinate
        z -- altitude
        """
        if rmax <= rmin:
            raise ValueError("rmin must be < rmax")
        if not log:
            r = np.linspace(rmin, rmax, nr)
        else:
            if rmin <= 0:
                raise ValueError("For log, rmin must be strictly >0")
            r = np.logspace(np.log10(rmin), np.log10(rmax), nr)
        if np.abs(theta) < 1e-6 and np.abs(phi) < 1e-6:
            x = r - self.x0
            y = 0
            z = 0
        else:
            x = r * np.sin(theta) * np.cos(phi) - self.x0
            y = r * np.sin(theta) * np.sin(theta) - self.y0
            z = r * np.cos(theta) - self.z0

        x, y, z, ax, ay, az = self.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        ax_ana, ay_ana, az_ana = self.cmpAcceleration_ana((x,y,z))
        if yscale == 'log':
            ax = np.abs(ax)
            ay = np.abs(ay)
            az = np.abs(az)
            ax_ana = np.abs(ax_ana)
            ay_ana = np.abs(ay_ana)
            az_ana = np.abs(az_ana)

        iplot = 1
        avec = [ax, ay, az]
        astr = ['$a_x$', '$a_y$', '$a_z$']
        while iplot < 4:
            if iplot == 1:
                axis = plt.subplot(311)
                locax = axis
            else:
                locax = plt.subplot(3,1,iplot, sharex = axis)
            v = avec[iplot-1]
            locax.plot(r, v, color = 'black')
            if iplot == 1:
                locax.plot(r, ax_ana, color = 'blue', linestyle = '--')
            elif iplot == 2:
                locax.plot(r, ay_ana, color = 'blue', linestyle = '--')
            elif iplot == 3:
                locax.plot(r, az_ana, color = 'blue', linestyle = '--')
            else:
                pass
                
            ylim = plt.ylim()
            if not log and r[0] < 0:
                locax.plot([-self.radius, -self.radius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            locax.plot([self.radius, self.radius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            if self.hollow:
                if not log and r[0] < 0:
                    plt.plot([-self.innerRadius, -self.innerRadius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
                plt.plot([self.innerRadius, self.innerRadius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            if log:
                locax.set_xscale('log')
            locax.set_yscale(yscale)
            #if iplot == 3:
            locax.set_xlabel('r (m)', fontsize = 13)
            #else:
            #    locax.set_xticklabels([])
            #locax.set_ylabel(astr[iplot-1] + r'$(r, z=' + str(z) + ')$ [$m.s^{-2}$]', fontsize = 13)
            locax.set_ylabel(astr[iplot-1], fontsize = 13)
            locax.tick_params(axis = 'both', which = 'major', labelsize = 13)
            iplot += 1
        plt.suptitle(r'$a(r, z=' + str(z) + ')$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0, hspace = 0.05)
        plt.show(block = True)


    def plt_aslice(self, xmin, xmax, ymin, ymax, z, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, ny = 200, nr = 100, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential in a horizontal slice (in (x,y) plane)
        """
        if z != 0:
            #ax, ay and az are non-zero, must compute them all. Could use a trick that I'm not eager to look for now... so do the slow, full numeric computation
            iplot = 1
            astr = ['$a_x$', '$a_y$', '$a_z$']
            comp = ['x', 'y', 'z']
            ax = None
            ay = None
            az = None
            while iplot < 4:
                if iplot == 1:
                    axis = plt.subplot(311)
                    locax = axis
                else:
                    locax = plt.subplot(3,1,iplot, sharex = axis)
                
                ax, ay, az = self.plt_aSlice('(x,y)', xmin, xmax, ymin, ymax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = (nx, ny), orthogonalCross = z, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, ax = ax, ay = ay, az = az)

                theta = np.linspace(0, 2*np.pi, 500)
                xc = self.radius * np.cos(theta)
                yc = self.radius * np.sin(theta)
                locax.plot(xc, yc, color = 'black')
                if self.hollow:
                    xc = self.innerRadius * np.cos(theta)
                    yc = self.innerRadius * np.sin(theta)
                    plt.plot(xc, yc, color = 'black')
                locax.set_xlim(xmin, xmax)
                locax.set_ylim(ymin, ymax)
                locax.set_ylabel('y [m]')
                if iplot == 3:
                    locax.set_xlabel('x [m]')
                else:
                    locax.set_xticklabels([])
                iplot += 1
            #plt.suptitle(r'$a(' + laborth+ ', z)$ [$m.s^{-2}$]', fontsize = 13)
            plt.subplots_adjust(wspace = 0, hspace = 0.05)
            plt.show(block = True)

        else:
            #in this case (z=0), az=0 and we only need to compute ax and ay, with an easy rotation invariance in the (x,y) plane
            #first, compute radially
            r = np.linspace(xmin, xmax, nr)
            x1 = r
            y1 = 0
            x1, y1, z1, ax1, ay1, az1 = self.a_grid(x1, y1, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)

            #then, rotation symmetry
            x = np.linspace(xmin, xmax, nx)
            y = np.linspace(ymin, ymax, ny)
            xp, yp = np.meshgrid(x, y)
            rp = np.sqrt(xp**2 + yp**2)
            thetap = np.arctan2(yp, xp)
            ax = np.zeros((nx,ny))
            ay = np.zeros((nx,ny))
            az = np.zeros((nx,ny))
            fx = interp1d(r, ax1, kind = 'linear', bounds_error = False, fill_value = 0)
            fy = interp1d(r, ay1, kind = 'linear', bounds_error = False, fill_value = 0)
            fz = interp1d(r, az1, kind = 'linear', bounds_error = False, fill_value = 0)
            #ax = np.cos(thetap) * fx(rp) - np.sin(thetap) * fy(rp)
            #ay = np.sin(thetap) * fx(rp) + np.cos(thetap) * fy(rp)
            ax = np.cos(thetap) * fx(rp) #fx(rp) is ax taken along the x-axis, i.e. the (signed) norm of the acceleration; this is exactly like the full rotation above, but with the symmetry assumption y->-y explicit
            ay = np.sin(thetap) * fx(rp)
            #careful to lines vs rows...
            ax = np.transpose(ax)
            ay = np.transpose(ay)
            az = np.transpose(az)

            i = 0
            ax0 = None
            for v in [ax, ay, az]:
                i += 1
                pltAxis = plt.subplot(3,1,i, sharex = ax0, sharey = ax0)
                if ax0 is None:
                    ax0 = pltAxis
                if i == 1:
                    _name = 'ax'
                    v1x = ax1
                    v1y = ay1
                elif i == 2:
                    _name = 'ay'
                    #v1 = ay1
                    v1x = ax1
                    v1y = ay1
                elif i == 3:
                    _name = 'az'
                    v1 = az1
                else:
                    raise ValueError('WTH!?!')
            
                if log:
                    v_slice = np.abs(v)
                    if i < 3:
                        v1xp = np.abs(v1x)
                        v1yp = np.abs(v1y)
                        _m = min(np.min(v1xp[v1xp > 0]), np.min(v1yp[v1yp > 0]))
                        _M = max(np.max(v1xp[v1xp > 0]), np.max(v1yp[v1yp > 0]))
                    else:
                        v1p = np.abs(v1)
                        _m = np.min(v1p[v1p > 0])
                        _M = np.max(v1p[v1p > 0])
                    if _m == 0:
                        print("WARNING! solid.plt_vatSlice: min value is 0 (colorbar). Setting it to 0.01 of max")
                        _m = 0.01 * _M
                        #_m = _av - 3 * _rms
                    nan = np.where((np.isfinite(v_slice) == False) | np.isnan(v_slice))
                    v_slice[nan] = _m
                    norm = LogNorm(vmin = _m, vmax = _M)
                    #norm = LogNorm(vmin = np.min(v_slice), vmax = np.max(v_slice))
                else:
                    v_slice = v
                    norm = None
                v_slice = np.flipud(np.transpose(v_slice)) #v_grid output coord#1 as lines and coord#2 as columns, but we want the opposite (thence transpose), and then imshow shows first lines on top, but we want them at the bottom, thence flipud
                im = pltAxis.imshow(v_slice, cmap = 'jet', aspect = 'auto', extent = [x[0], x[-1], y[0], y[-1]], norm = norm)
                theta = np.linspace(0, 2*np.pi, 500)
                xc = self.radius * np.cos(theta)
                yc = self.radius * np.sin(theta)
                pltAxis.plot(xc, yc, color = 'black', linestyle = '--')
                if self.hollow:
                    xc_in = self.innerRadius * np.cos(theta)
                    yc_in = self.innerRadius * np.sin(theta)
                    pltAxis.plot(xc_in, yc_in, color = 'black', linestyle = '--')
                cbar = plt.colorbar(im)
                cbar.ax.set_ylabel(_name + ' [m/s$^2$]')
                pltAxis.set_xlabel('x')
                pltAxis.set_ylabel('y')
                #pltAxis.set_title("Potential " + axis3 + "=" + str(orthogonalCross))
                if i == 1:
                    pltAxis.set_title("Acceleration z=" + str(z))
            plt.show(block = True)


def pltVr(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = False, yscale = 'linear', nr = 30, origin = [0.,0.,0.]):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 30e-2 #m
    innerRadius = 25e-2 #m
    density = 2830 #kg/m^3
    b = ball(radius, density = density, innerRadius = innerRadius, origin = origin)
    b.plt_vr(-3, 3, nr = nr, theta = 0, phi = 0, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

    
def pltAr(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = False, yscale = 'linear', nr = 30, origin = [0.,0.,0.]):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 30e-2 #m
    innerRadius = 25e-2 #m
    density = 2830 #kg/m^3
    b = ball(radius, density = density, innerRadius = innerRadius, origin = origin)
    b.plt_ar(-2, 2, nr = nr, theta = 0, phi = 0, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltAh_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, ny = 15, nr = 30, log = True, origin = [0.,0.,0.]):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 30e-2 #m
    innerRadius = 25e-2 #m
    density = 2830 #kg/m^3
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1
    z = 0
    b = ball(radius, density = density, innerRadius = innerRadius, origin = origin)
    b.plt_aslice(xmin, xmax, ymin, ymax, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, nx = nx, ny = ny, nr = nr, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)


def inertiaMoments(fullNumeric = False):
    b = ball(1.3, density = 1, innerRadius = 0)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                b.inertiaMoment(i,j,k, dx_rel = 0.005, dy_rel = 0.005, dz_rel = 0.01, verbose = False, fullNumeric = fullNumeric)
                print(i,j,k, getattr(b, 'I' + str(i) + str(j) + str(k)))
