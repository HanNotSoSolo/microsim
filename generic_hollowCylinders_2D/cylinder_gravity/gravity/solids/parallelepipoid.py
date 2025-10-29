import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
#try:
#    from skmonaco import mcquad #scikit-monaco
#except:
#    print("WARNING! scikit-monaco not found, and seems obsolete. Cannot perform Monte Carlo integration (or need a new package)")
from scipy.constants import G
from scipy.integrate import tplquad
import time
from gravity.solids import solid

class parallelepipoid(solid.solid):
    def __init__(self, length, width, height, origin = [0.,0.,0.], density = 1, originType = 'centered'):
        super(parallelepipoid, self).__init__(density, origin = origin)
        self.length = length #along x
        self.width = width #along y
        self.height = height #along z
        if originType == 'centered':
            self.hmin = self.z0 - 0.5 * height
            self.hmax = self.z0 + 0.5 * height
        elif originType == 'low':
            self.hmin = self.z0
            self.hmax = height
        else:
            raise NotImplementedError()
        self.xmin = self.x0 - 0.5 * self.length #lower boundary for x for tplquad integration
        self.xmax = self.x0 + 0.5 * self.length #upper boundary for x for tplquad integration
        self.ymin = self.y0 - 0.5 * self.width #lower boundary for x for tplquad integration
        self.ymax = self.y0 + 0.5 * self.width #upper boundary for x for tplquad integration
        
        self.tplquad_xmin = self.xmin #lower boundary for x for tplquad integration
        self.tplquad_xmax = self.xmax #upper boundary for x for tplquad integration

    def tplquad_ymin_fun(self, x):
        """lower boundary for y for tplquad integration (gfun in terms of tplquad doc)"""
        return self.ymin

    def tplquad_ymax_fun(self, x):
        """upper boundary for y for tplquad integration (hfun in terms of tplquad doc)"""
        return self.ymax

    def tplquad_zmin_fun(self, x, y):
        """lower boundary for z for tplquad integration (qfun in terms of tplquad doc)"""
        return self.hmin

    def tplquad_zmax_fun(self, x, y):
        """upper boundary for z for tplquad integration (rfun in terms of tplquad doc)"""
        return self.hmax
        
    def cmpVolume(self):
        return self.length * self.width * self.height

    def inside(self, xyzmin, xyzmax, strictBorder = True):
        """
        Check if a small volume (or an ensemble of small volumes) is inside or outside the parallelepipoid. Return 1 for volume(s) inside the parallelepipoid, 0 for others
        xyzmin -- (xleft, yleft, zmin)
        xyzmax -- (xright, yright, zmax)
        strictBorder -- if set, a small cube is said to be inside the parallelepipoid if it is entirely inside. Otherwise, said to be inside even if it is halfway out.
        """
        xleft = xyzmin[0] - self.x0
        yleft = xyzmin[1] - self.y0
        zmin = xyzmin[2] - self.z0
        xright = xyzmax[0] - self.x0
        yright = xyzmax[1] - self.y0
        zmax = xyzmax[2] - self.z0

        if np.size(xleft) == 1:
            if strictBorder:
                if xright <= self.xmax and xleft >= self.xmin and yright <= self.ymax and yleft >= self.ymin and zmax <= self.hmax and zmin >= self.hmin:
                    return 1
                else:
                    return 0
            else:
                if ((xleft <= self.xmax and xleft >= self.xmin) or (xright >= self.xmin and xright <= self.xmax)) and ((yleft <= self.ymax and yleft >= self.ymin) or (yright >= self.ymin and yright <= self.ymax)) and ((zmin <= self.hmax and zmin >= self.hmin) or (zmax >= self.hmin and zmax <= self.hmax)):
                    return 1
                else:
                    return 0
        else:
            output = np.zeros(np.shape(xleft))
            if strictBorder:
                inner = np.where((xright <= self.xmax) & (xleft >= self.xmin) & (yright <= self.ymax) & (yleft >= self.ymin) & (zmax <= self.hmax) & (zmin >= self.hmin))#[0]
            else:
                inner = np.where((((xleft <= self.xmax) & (xleft >= self.xmin)) | ((xright >= self.xmin) & (xright <= self.xmax))) & (((yleft <= self.ymax) & (yleft >= self.ymin)) | ((yright >= self.ymin) & (yright <= self.ymax))) & (((zmin <= self.hmax) & (zmin >= self.hmin)) | ((zmax >= self.hmin) & (zmax <= self.hmax))))#[0]
            output[inner] = 1
            return output
            
    def insidePoint(self, x_in, y_in, z_in):
        """
        Check if a point (or an ensemble of points) is inside or outside the parallelepipoid. Return 1 for point(s) inside the parallelepipoid, 0 for others
        """
        x = x_in - self.x0
        y = y_in - self.y0
        z = z_in - self.z0
        if np.size(x) == 1:
            if x <= self.xmax and x >= self.xmin and y <= self.ymax and y >= self.ymin and z <= self.hmax and z >= self.hmin:
                return 1
            else:
                return 0
        else:
            output = np.zeros(np.shape(x))
            inner = np.where((x <= self.xmax) & (x >= self.xmin) & (y <= self.ymax) & (y >= self.ymin) & (z <= self.hmax) & (z >= self.hmax))#[0]
            output[inner] = 1
            return output

    def mkGrid(self, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, borders = True):
        """
        Make regular grid inside parallelepipoid
        dx, dy, dz -- relative size (wrt to parallelepipoid size) of volume elements
        """
        dx = self.length * dx_rel
        dy = self.width * dy_rel
        dz = self.height * dz_rel

        xspan = self.xmax + 0.05 * self.length - (self.xmin - 0.05 * self.length)
        nx = int(xspan / dx) + 1
        x = np.linspace(self.xmin - 0.05 * self.length, self.xmax + 0.05 * self.length, nx) + self.x0
        yspan = self.ymax + 0.05 * self.width - (self.ymin - 0.05 * self.width)
        ny = int(yspan / dy) + 1
        y = np.linspace(self.ymin - 0.05 * self.width, self.ymax + 0.05 * self.width, ny) + self.y0
        zspan = self.hmax + 0.05 * self.height - (self.hmin - 0.05 * self.height)
        nz = int(zspan / dz) + 1
        z = np.linspace(self.hmin - 0.05 * self.height, self.hmax + 0.05 * self.height, nz)
        
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
        

    def inertiaMoment(self, i, j, k, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, method = 'simps', mc_npoints = 1e6, memoryFriendly = False, verbose = True):
        """
        Compute ijk inertia moment (Int[rho*x^i*y^j*z^k dxdydz])
        don't care about arguments other than i,j,k (here just to be consistent with signature of method in solid class)
        """
        xterm = (self.xmax**(i+1) - self.xmin**(i+1)) / (i+1)
        yterm = (self.ymax**(j+1) - self.ymin**(j+1)) / (j+1)
        zterm = (self.zmax**(k+1) - self.zmin**(k+1)) / (k+1)
        I = xterm * yterm * zterm

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
        #X = XYZ[0]
        #Y = XYZ[1]
        #Z = XYZ[2]
        #
        #if timeit:
        #    st = time.time()
        #dU = lambda xyz: self.infinitesimalVElementPotential(xyz[0], xyz[1], xyz[2], X, Y, Z)
        #U, err = mcquad(dU, npoints, [self.radius, self.radius, 0.5 * self.height], [-self.radius, -self.radius, -0.5 * self.height])
        #if np.abs(err) >= 1e-3 * np.abs(U):
        #    print("WARNING! mcquad error is more than 1e-3 integral estimate...", U, err)
        #U *= G
        #if timeit:
        #    et = time.time()
        #    return (U, et - st)
        #return U

    # def cmpPotential_XYZ_inertiaMoments(self, XYZ, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, im_method = 'simps', memoryFriendly = False, timeit = False):
    #     """
    #     XYZ -- (X, Y, Z)
    #     memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
    #     """
    #     raise NotImplementedError("Gotta to do the maths")
    #     X = XYZ[0]
    #     Y = XYZ[1]
    #     Z = XYZ[2]
        
    #     if timeit:
    #         st = time.time()
    #     for i in range(4):
    #         for j in range(4):
    #             for k in range(4):
    #                 if i + j + k == 0:
    #                     continue
    #                 if i + j + k > 3:
    #                     continue
    #                 ilocal = getattr(self, 'I' + str(i) + str(j) + str(k))
    #                 if ilocal is None:
    #                     self.inertiaMoment(i, j, k, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, method = im_method, memoryFriendly = memoryFriendly)
    #     mass = self.cmpMass()
    #     R = np.sqrt(X**2 + Y**2 + Z**2)
    #     #V0 = ...
    #     #V3 = ...
    #     #V5 = ...
    #     #V7 = ...
    #     #V = G * (V0 + V3 + V5 + V7)
    #     ##V = G * V0 #Gm/R
    #     #if timeit:
    #     #    et = time.time()
    #     #    return (V, et - st)
    #     #return V
    


    def plt_vAxis(self, axis, _min, _max, n = 100, cross1 = 0, cross2 = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential as a function of distance (parallel to axis)
        axis -- x, y or z
        _min, _max - x,y,z limits [m]
        cross1, cross2 -- other coordinates where potential is plotted
        """
        if axis == 'x':
            _m = self.xmin
            _M = self.xmax
            xlabel = 'x [m]'
            laborth = "y=" + str(cross1) + ", z=" + str(cross2)
            ylabel = r'$V(' + laborth+ ', x)$ [$J.kg^{-1}$]'
        elif axis == 'y':
            _m = self.ymin
            _M = self.ymax
            xlabel = 'y [m]'
            laborth = "x=" + str(cross1) + ", z=" + str(cross2)
            ylabel = r'$V(' + laborth+ ', y)$ [$J.kg^{-1}$]'
        elif axis == 'z':
            _m = self.hmin
            _M = self.hmax
            xlabel = 'z [m]'
            laborth = "x=" + str(cross1) + ", y=" + str(cross2)
            ylabel = r'$V(' + laborth+ ', z)$ [$J.kg^{-1}$]'
        else:
            raise ValueError('Bad axis')
                        
        x, v = self.plt_vAlongAxis(axis, _min, _max, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, n = n, orthogonalPlaneCross = (cross1, cross2), log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True)
        ylim = plt.ylim()
        if not log and _m <= 0:
            plt.plot([_m, _m], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        plt.plot([_M, _M], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        if log:
            plt.xscale('log')
        plt.yscale(yscale)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show(block = True)
        
    def plt_vx(self, xmin, xmax, nx = 100, ycross = 0, zcross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential as a function of "length" (parallel to axis)
        xmin, xmax - x limits [m]
        ycross, zcross -- y and z where potential is plotted
        """
        self.plt_vAxis('x', xmin, xmax, n = nx, cross1 = ycross, cross2 = zcross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        

    def plt_vy(self, ymin, ymax, ny = 100, xcross = 0, zcross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential as a function of depth (parallel to axis)
        ymin, ymax - y limits [m]
        xcross, zcross -- x and z where potential is plotted
        """
        self.plt_vAxis('y', ymin, ymax, n = ny, cross1 = xcross, cross2 = zcross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        
    def plt_vz(self, zmin, zmax, nz = 100, xcross = 0, ycross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential as a function of height (parallel to axis)
        zmin, zmax - z limits [m]
        xcross, ycross -- x and y where potential is plotted
        """
        self.plt_vAxis('z', zmin, zmax, n = nz, cross1 = xcross, cross2 = ycross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        

    def plt_aAxis(self, axis, _min, _max, n = 100, cross1 = 0, cross2 = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot acceleration as a function of distance (parallel to axis)
        _min, _max - x,yz limits [m]
        cross1, cross2 -- other coordinates where potential is plotted
        """
        if axis == 'x':
            _m = self.xmin
            _M = self.xmax
            xlabel = 'x [m]'
            laborth = "y=" + str(cross1) + ", z=" + str(cross2)
            suptitle = r'$a(' + laborth+ ', x)$ [$m.s^{-2}$]'
        elif axis == 'y':
            _m = self.ymin
            _M = self.ymax
            xlabel = 'y [m]'
            laborth = "x=" + str(cross1) + ", z=" + str(cross2)
            suptitle = r'$a(' + laborth+ ', y)$ [$m.s^{-2}$]'
        elif axis == 'z':
            _m = self.hmin
            _M = self.hmax
            xlabel = 'z [m]'
            laborth = "x=" + str(cross1) + ", y=" + str(cross2)
            suptitle = r'$a(' + laborth+ ', z)$ [$m.s^{-2}$]'
        else:
            raise ValueError('Bad axis')

        iplot = 1
        astr = ['$a_x$', '$a_y$', '$a_z$']
        comp = ['x', 'y', 'z']
        ax = None
        ay = None
        az = None
        while iplot < 4:
            if iplot == 1:
                axis_tmp = plt.subplot(311)
                locax = axis_tmp
            else:
                locax = plt.subplot(3,1,iplot, sharex = axis_tmp)
            z, ax, ay, az = self.plt_aAlongAxis(axis, _min, _max, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = n, orthogonalPlaneCross = (cross1, cross2), log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, ax = ax, ay = ay, az = az)
            ylim = plt.ylim()
            if not log and _m:
                locax.plot([_m, _m], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            locax.plot([_M, _M], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            if log:
                locax.set_xscale('log')
            locax.set_yscale(yscale)
            locax.set_ylabel(astr[iplot-1], fontsize = 13)
            if iplot == 3:
                locax.set_xlabel(xlabel)
            else:
                locax.set_xticklabels([])
            iplot += 1
        plt.suptitle(suptitle, fontsize = 13)
        plt.subplots_adjust(wspace = 0, hspace = 0.05)
        plt.show(block = True)

    def plt_TAxis(self, axis, _min, _max, n = 100, cross1 = 0, cross2 = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot GGT as a function of distance (parallel to axis)
        _min, _max - x,yz limits [m]
        cross1, cross2 -- other coordinates where potential is plotted
        """
        if axis == 'x':
            _m = self.xmin
            _M = self.xmax
            xlabel = 'x [m]'
            laborth = "y=" + str(cross1) + ", z=" + str(cross2)
            suptitle = r'$T(' + laborth+ ', x)$ [$s^{-2}$]'
        elif axis == 'y':
            _m = self.ymin
            _M = self.ymax
            xlabel = 'y [m]'
            laborth = "x=" + str(cross1) + ", z=" + str(cross2)
            suptitle = r'$T(' + laborth+ ', y)$ [$s^{-2}$]'
        elif axis == 'z':
            _m = self.hmin
            _M = self.hmax
            xlabel = 'z [m]'
            laborth = "x=" + str(cross1) + ", y=" + str(cross2)
            suptitle = r'$T(' + laborth+ ', z)$ [$s^{-2}$]'
        else:
            raise ValueError('Bad axis')

        iplot = 1
        astr = ['$T_{xx}$', '$T_{xy}$', '$T_{xz}$', '$T_{yx}$', '$T_{yy}$', '$T_{yz}$', '$T_{zx}$', '$T_{zy}$', '$T_{zz}$']
        comp = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        Txx = None
        Txy = None
        Txz = None
        Tyy = None
        Tyz = None
        Tzz = None
        while iplot < 10:
            if iplot in [4,7,8]:
                iplot += 1
                continue
            if iplot == 1:
                axis_tmp = plt.subplot(331)
                locax = axis_tmp
            else:
                locax = plt.subplot(3,3,iplot, sharex = axis_tmp)
            z, Txx, Txy, Txz, Tyy, Tyz, Tzz = self.plt_TAlongAxis(axis, _min, _max, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, Tcomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = n, orthogonalPlaneCross = (cross1, cross2), log = log, yscale = yscale, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, verbose = verbose, hold = True, Txx = Txx, Txy = Txy, Txz = Txz, Tyy = Tyy, Tyz = Tyz, Tzz = Tzz)
            ylim = plt.ylim()
            if not log and _m:
                locax.plot([_m, _m], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            locax.plot([_M, _M], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            if log:
                locax.set_xscale('log')
            locax.set_yscale(yscale)
            locax.set_ylabel(astr[iplot-1], fontsize = 13)
            if iplot in [1,5,9]:
                locax.set_xlabel(xlabel)
            else:
                locax.set_xticklabels([])
            iplot += 1
        plt.suptitle(suptitle, fontsize = 13)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        plt.show(block = True)

    def plt_ax(self, xmin, xmax, nx = 100, ycross = 0, zcross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot acceleration as a function of distance (parallel to x-axis)
        xmin, xmax - x limits [m]
        ycross, zcross -- y and z where potential is plotted
        """
        self.plt_aAxis('x', xmin, xmax, n = nx, cross1 = ycross, cross2 = zcross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)

    def plt_ay(self, ymin, ymax, ny = 100, xcross = 0, zcross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot acceleration as a function of distance (parallel to y-axis)
        ymin, ymax - y limits [m]
        xcross, zcross -- x and z where potential is plotted
        """
        self.plt_aAxis('y', ymin, ymax, n = ny, cross1 = xcross, cross2 = zcross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        
    def plt_az(self, zmin, zmax, nz = 100, xcross = 0, ycross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot acceleration as a function of distance (parallel to z-axis)
        zmin, zmax - z limits [m]
        xcross, ycross -- x and y where potential is plotted
        """
        self.plt_aAxis('z', zmin, zmax, n = nz, cross1 = xcross, cross2 = ycross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)

    def plt_Tx(self, xmin, xmax, nx = 100, ycross = 0, zcross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot GGT as a function of distance (parallel to x-axis)
        xmin, xmax - x limits [m]
        ycross, zcross -- y and z where potential is plotted
        """
        self.plt_TAxis('x', xmin, xmax, n = nx, cross1 = ycross, cross2 = zcross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)

    def plt_Ty(self, ymin, ymax, ny = 100, xcross = 0, zcross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot GGT as a function of distance (parallel to y-axis)
        ymin, ymax - y limits [m]
        xcross, zcross -- x and z where potential is plotted
        """
        self.plt_TAxis('y', ymin, ymax, n = ny, cross1 = xcross, cross2 = zcross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        
    def plt_Tz(self, zmin, zmax, nz = 100, xcross = 0, ycross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False):
        """
        Plot acceleration as a function of distance (parallel to z-axis)
        zmin, zmax - z limits [m]
        xcross, ycross -- x and y where potential is plotted
        """
        self.plt_TAxis('z', zmin, zmax, n = nz, cross1 = xcross, cross2 = ycross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, log = log, yscale = yscale, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        
    def plt_transversal_vslice(self, plane, min1, max1, min2, max2, cross, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, n1 = 200, n2 = 200, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        self.plt_transversal_vrhoslice('potential', plane, min1, max1, min2, max2, cross, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, n1 = n1, n2 = n2, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints =  mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
        
    def plt_transversal_vrhoslice(self, _type, plane, min1, max1, min2, max2, cross, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, n1 = 200, n2 = 200, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential or density in a transversale (horizontal or vertical) slice
        plane -- (x,y) or (x,z) or (y,z)
        cross -- z or y or x where plane crosses orthogonal direction
        """
        if _type == 'potential':
            self.plt_vSlice(plane, min1, max1, min2, max2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, n = (n1, n2), orthogonalCross = cross, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True)
        elif _type == 'density':
            self.plt_rhoSlice(plane, min1, max1, min2, max2, n = (n1, n2), orthogonalCross = cross, hold = True)
        else:
            raise ValueError('Bad type')

        if plane == '(x,y)':
            m1 = self.xmin
            M1 = self.xmax
            m2 = self.ymin
            M2 = self.ymax
            mc = self.hmin
            Mc = self.hmax
            xlabel = 'x [m]'
            ylabel = 'y [m]'
        elif plane == '(x,z)':
            m1 = self.xmin
            M1 = self.xmax
            m2 = self.hmin
            M2 = self.hmax
            mc = self.ymin
            Mc = self.ymax
            xlabel = 'x [m]'
            ylabel = 'z [m]'
        elif plane == '(y,z)':
            m1 = self.ymin
            M1 = self.ymax
            m2 = self.hmin
            M2 = self.hmax
            mc = self.xmin
            Mc = self.xmax
            xlabel = 'y [m]'
            ylabel = 'z [m]'
        else:
            raise ValueError('Bad plane')
        
        if cross <= Mc and cross >= mc:
            linestyle = '--'
        else:
            linestyle = ':'
        plt.plot([m1, M1], [M2, M2], color = 'black', linestyle = linestyle)
        plt.plot([m1, M1], [m2, m2], color = 'black', linestyle = linestyle)
        plt.plot([m1, m1], [m2, M2], color = 'black', linestyle = linestyle)
        plt.plot([M1, M1], [m2, M2], color = 'black', linestyle = linestyle)
        plt.xlim(min1, max1)
        plt.ylim(min2, max2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show(block = True)
        

    def plt_transversal_aslice(self, plane, min1, max1, min2, max2, cross, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, n1 = 200, n2 = 200, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot acceleration in a transversale (horizontal or vertical) slice
        plane -- (x,y) or (x,z) or (y,z)
        cross -- z or y or x where plane crosses orthogonal direction
        """
        if plane == '(x,y)':
            m1 = self.xmin
            M1 = self.xmax
            m2 = self.ymin
            M2 = self.ymax
            mc = self.hmin
            Mc = self.hmax
            xlabel = 'x [m]'
            ylabel = 'y [m]'
        elif plane == '(x,z)':
            m1 = self.xmin
            M1 = self.xmax
            m2 = self.hmin
            M2 = self.hmax
            mc = self.ymin
            Mc = self.ymax
            xlabel = 'x [m]'
            ylabel = 'z [m]'
        elif plane == '(y,z)':
            m1 = self.ymin
            M1 = self.ymax
            m2 = self.hmin
            M2 = self.hmax
            mc = self.xmin
            Mc = self.xmax
            xlabel = 'y [m]'
            ylabel = 'z [m]'
        else:
            raise ValueError('Bad plane')

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
                
            ax, ay, az = self.plt_aSlice(plane, min1, max1, min2, max2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = (n1, n2), orthogonalCross = cross, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, ax = ax, ay = ay, az = az)

            if cross <= Mc and cross >= mc:
                linestyle = '--'
            else:
                linestyle = ':'
            locax.plot([m1, M1], [M2, M2], color = 'black', linestyle = linestyle)
            locax.plot([m1, M1], [m2, m2], color = 'black', linestyle = linestyle)
            locax.plot([m1, m1], [m2, M2], color = 'black', linestyle = linestyle)
            locax.plot([M1, M1], [m2, M2], color = 'black', linestyle = linestyle)
            locax.set_xlim(min1, max1)
            locax.set_ylim(min2, max2)
            locax.set_ylabel(ylabel)
            if iplot == 3:
                locax.set_xlabel(xlabel)
            else:
                locax.set_xticklabels([])
            iplot += 1
        #plt.suptitle(r'$a(' + laborth+ ', z)$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0, hspace = 0.05)
        plt.show(block = True)


    def plt_transversal_Tslice(self, plane, min1, max1, min2, max2, cross, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, n1 = 200, n2 = 200, log = False, v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False):
        """
        Plot GGT in a transversale (horizontal or vertical) slice
        plane -- (x,y) or (x,z) or (y,z)
        cross -- z or y or x where plane crosses orthogonal direction
        """
        if plane == '(x,y)':
            m1 = self.xmin
            M1 = self.xmax
            m2 = self.ymin
            M2 = self.ymax
            mc = self.hmin
            Mc = self.hmax
            xlabel = 'x [m]'
            ylabel = 'y [m]'
        elif plane == '(x,z)':
            m1 = self.xmin
            M1 = self.xmax
            m2 = self.hmin
            M2 = self.hmax
            mc = self.ymin
            Mc = self.ymax
            xlabel = 'x [m]'
            ylabel = 'z [m]'
        elif plane == '(y,z)':
            m1 = self.ymin
            M1 = self.ymax
            m2 = self.hmin
            M2 = self.hmax
            mc = self.xmin
            Mc = self.xmax
            xlabel = 'y [m]'
            ylabel = 'z [m]'
        else:
            raise ValueError('Bad plane')

        iplot = 1
        astr = ['$T_{xx}$', '$T_{xy}$', '$T_{xz}$', '$T_{yx}$', '$T_{yy}$', '$T_{yz}$', '$T_{zx}$', '$T_{zy}$', '$T_{zz}$']
        comp = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        Txx = None
        Txy = None
        Txz = None
        Tyy = None
        Tyz = None
        Tzz = None
        while iplot < 10:
            if iplot in [4,7,8]:
                iplot += 1
                continue
            if iplot == 1:
                axis = plt.subplot(331)
                locax = axis
            else:
                locax = plt.subplot(3,3,iplot, sharex = axis)

            Txx, Txy, Txz, Tyy, Tyz, Tzz = self.plt_TSlice(plane, min1, max1, min2, max2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, Tcomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = (n1, n2), orthogonalCross = cross, log = log, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, Txx = Txx, Txy = Txy, Txz = Txz, Tyy = Tyy, Tyz = Tyz, Tzz = Tzz)

            if cross <= Mc and cross >= mc:
                linestyle = '--'
            else:
                linestyle = ':'
            locax.plot([m1, M1], [M2, M2], color = 'black', linestyle = linestyle)
            locax.plot([m1, M1], [m2, m2], color = 'black', linestyle = linestyle)
            locax.plot([m1, m1], [m2, M2], color = 'black', linestyle = linestyle)
            locax.plot([M1, M1], [m2, M2], color = 'black', linestyle = linestyle)
            locax.set_xlim(min1, max1)
            locax.set_ylim(min2, max2)
            locax.set_ylabel(ylabel)
            if iplot in [1,5,9]:
                locax.set_xlabel(xlabel)
            else:
                locax.set_xticklabels([])
            iplot += 1
        #plt.suptitle(r'$a(' + laborth+ ', z)$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        plt.show(block = True)

    def plt_transversal_rhoslice(self, plane, min1, max1, min2, max2, cross, objectRef = True, n1 = 200, n2 = 200, log = False, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, verbose = False):
        self.plt_transversal_vrhoslice('density', plane, min1, max1, min2, max2, cross, objectRef = objectRef, n1 = n1, n2 = n2, log = log, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, verbose = verbose)
        


        




#############################################
def pltVx(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    length = 3.8
    width = 2.4
    height = 1.6
    density = 1
    p = parallelepipoid(length, width, height, density = density, originType = originType)
    p.plt_vx(-5, 5, nx = 120, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, yscale = yscale)


def pltVz(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    length = 3.8
    width = 2.4
    height = 1.6
    density = 1
    p = parallelepipoid(length, width, height, density = density, originType = originType)
    p.plt_vz(0.1, 35, nz = 120, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)
    p.plt_vz(0.1, 35, nz = 120, xcross = 1.3, ycross = 0, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, yscale = yscale)

    
def pltVh_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, ny = 15, log = True):
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    xmin = -1600
    xmax = 1600
    ymin = -1350
    ymax = 1350
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_transversal_vslice('(x,y)', xmin, xmax, ymin, ymax, height / 2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = True, n1 = nx, n2 = ny, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False)
    pa.plt_transversal_vslice('(x,y)', xmin, xmax, ymin, ymax, 2*height, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = True, n1 = nx, n2 = ny, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False)

    
def pltVv_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, nz = 15, log = True):
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    xmin = -1600
    xmax = 1600
    zmin = 150
    zmax = 850
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_transversal_vslice('(x,z)', xmin, xmax, zmin, zmax, 0, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = True, n1 = nx, n2 = nz, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False)


def pltAx(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_ax(170, 1000, nx = 120, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)
    
def pltAz(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_az(170, 1000, nz = 120, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)
    pa.plt_az(170, 1000, nz = 120, xcross = 1.3, ycross = 0, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)


def pltAh_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, ny = 15, log = True):
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    xmin = -1600
    xmax = 1600
    ymin = -1350
    ymax = 1350
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_transversal_aslice('(x,y)', xmin, xmax, ymin, ymax, height / 2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = True, n1 = nx, n2 = ny, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False)
    pa.plt_transversal_aslice('(x,y)', xmin, xmax, ymin, ymax, 2*height, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = True, n1 = nx, n2 = ny, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False)

def pltAv_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, nz = 15, log = True):
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    xmin = -1600
    xmax = 1600
    zmin = 170
    zmax = 700
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_transversal_aslice('(x,z)', xmin, xmax, zmin, zmax, 0, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = True, n1 = nx, n2 = nz, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False)

def pltTx(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_Tx(170, 1000, nx = 60, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)

def pltTz(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_Tz(170, 1000, nz = 120, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)


def pltTh_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, ny = 15, log = True):
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    xmin = -1600
    xmax = 1600
    ymin = -1350
    ymax = 1350
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    #pa.plt_transversal_Tslice('(x,y)', xmin, xmax, ymin, ymax, height / 2, objectRef = True, n1 = nx, n2 = ny, log = log, v_method = v_method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = False, verbose = False)
    pa.plt_transversal_Tslice('(x,y)', xmin, xmax, ymin, ymax, 2*height, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = True, n1 = nx, n2 = ny, log = log, v_method = v_method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = False, verbose = False)
    
def pltRhoh_slice(nx = 15, ny = 15):
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    xmin = -600
    xmax = 600
    ymin = -350
    ymax = 350
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_transversal_rhoslice('(x,y)', xmin, xmax, ymin, ymax, height / 2, n1 = nx, n2 = ny)

def pltRhov_slice(nx = 15, nz = 15):
    length = 1000 #m
    width = 500 #m
    height = 200 #m
    density = 1000 #kg/m^3
    xmin = -600
    xmax = 600
    zmin = -50
    zmax = 250
    pa = parallelepipoid(length, width, height, density = density, originType = 'low')
    pa.plt_transversal_rhoslice('(x,z)', xmin, xmax, zmin, zmax, 0, n1 = nx, n2 = nz)
 
