import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.integrate import simpson
from scipy.interpolate import interp1d
#from skmonaco import mcquad #scikit-monaco
from scipy.constants import G
from scipy.integrate import tplquad
import numdifftools as nd
import time
from gravity.solids.progressETA import progress
from gravity.solids import plots

class solid(object):
    def __init__(self, density, center = [0.,0.,0.], origin = [0.,0.,0.]):
        self.kind = 'undefined'
        self.density = density
        #self.x0 = origin[0]
        #self.y0 = origin[1]
        #self.z0 = origin[2]
        #self.originType = None
        self.setCenter(center)
        self.setOrigin(origin)
        self.I100 = None
        self.I010 = None
        self.I001 = None
        self.I110 = None
        self.I101 = None
        self.I011 = None
        self.I200 = None
        self.I020 = None
        self.I002 = None
        self.I210 = None
        self.I201 = None
        self.I120 = None
        self.I021 = None
        self.I102 = None
        self.I012 = None
        self.I111 = None
        self.I300 = None
        self.I030 = None
        self.I003 = None
        self.prevProp = 0

    def setCenter(self, center):
        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]
        
    def setOrigin(self, origin):
        self.x0 = self.xc + origin[0]
        self.y0 = self.yc + origin[1]
        self.z0 = self.zc + origin[2]
        
    def inner(self, xyzmin, xyzmax, strictBorder = True):
        raise NotImplementedError("Now called inside()")
        
    def inside(self, xyzmin, xyzmax, strictBorder = True):
        """
        Check if a small volume (or an ensemble of small volumes) is inside or outside the cylinder. Return 1 for volume(s) inside the cylinder, 0 for others
        xyzmin -- (xleft, yleft, zmin)
        xyzmax -- (xright, yright, zmax)
        strictBorder -- if set, a small cube is said to be inside the cylinder if it is entirely inside. Otherwise, said to be inside even if it is halfway out.
        """
        raise NotImplementedError("Must be implemented in daughter classes")

    def innerPoint(self, x_in, y_in, z_in):
        raise NotImplementedError("Now called insidePoint()")
        
    def insidePoint(self, x_in, y_in, z_in):
        """
        Check if a point (or an ensemble of points) is inside or outside the cylinder. Return 1 for point(s) inside the cylinder, 0 for others
        """
        raise NotImplementedError("Must be implemented in daughter classes")
    
    def cmpVolume(self):
        raise NotImplementedError("Must be implemented in daughter classes")

    def cmpMass(self):
        return self.density * self.cmpVolume()

    def cmpMassNumeric(self, gridType = None, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01):
        """
        Compute potential created by infinitesimal volume element (xleft,yleft,zmin)-(xright,yright,zmax) at point (X,Y,Z) --WARNING! this is potential per volume, to accomodate simps integration
        """
        x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = self.mkGrid(dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, _type = gridType)

        yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
        yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
        xyzmin = (xl3, yl3, zl3)
        xyzmax = (xr3, yr3, zr3)
        #dU = self.vElementPotential((xl3, yl3, zl3), (xr3, yr3, zr3), (X, Y, Z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, gridType = gridType)
        
        # if self.kind == 'cylinder' and gridType == 'polar':
        #     #in this case, the grid is centered on (0,0,z0), so need to translate x and y to cbe centered on the cylinder
        #     rleft = xyzmin[0]
        #     thleft = xyzmin[1]
        #     rright = xyzmax[0]
        #     thright = xyzmax[1]
        #     xleft = rleft * np.cos(thleft) + self.x0
        #     yleft = rleft * np.sin(thleft) + self.y0
        #     xright = rright * np.cos(thright) + self.x0
        #     yright = rright * np.sin(thright) + self.y0
        #     zmin = xyzmin[2] #- self.z0
        #     zmax = xyzmax[2] #- self.z0
        # else:
        #     xleft = xyzmin[0]
        #     yleft = xyzmin[1]
        #     zmin = xyzmin[2]
        #     xright = xyzmax[0]
        #     yright = xyzmax[1]
        #     zmax = xyzmax[2]
        
        rho = self.vElementRho(xyzmin, xyzmax, verbose = False, gridType = gridType)
        #print(np.sum(rho))
        #print(np.min(xl3), np.min(yl3), np.min(zl3))
        #print(np.max(xr3), np.max(yr3), np.max(zr3))
        rho_xy = simps(rho, x = z, axis = 2)
        fig = plt.figure()
        ax = fig.add_subplot(111, polar = (gridType == 'polar'))
        #plt.imshow(rho_xy)
        #im = ax.pcolormesh(y, x, rho_xy)
        #fig.colorbar(im)
        #plt.show()
        if gridType == 'Cartesian':
            rho_x = simps(rho_xy, x = y, axis = 1)
            M = simps(rho_x, x = x, axis = 0)
        elif gridType == 'polar':
            rho_x = simps(rho_xy, x = y, axis = 1)
            M = simps(x * rho_x, x = x, axis = 0)
        else:
            raise ValueError('?!?')
        return M
        
    
    def vElementRho(self, xyzmin, xyzmax, verbose = False, gridType = None):
        """
        Get density in an elementary volume
        xyzmin -- (xleft, yleft, zmin)
        xyzmax -- (xright, yright, zmax)
        """
        inner = self.inside(xyzmin, xyzmax, strictBorder = True, gridType = gridType)
        return inner * self.density
    
    def infinitesimalVElementRho(self, x, y,z, gridType = None):
        """Same thing as vElementRho, but for dV->0, ie just a point"""
        inner = self.insidePoint(x, y, z, gridType = gridType)
        return inner * self.density
    
    def amrFunc(self, xl, xr, yl, yr, zl, zr, nx, ny, nz):
        print("solid.amrFunc: Work on AMR...")
        print(np.shape(xl), np.shape(yl), np.shape(zl))
        n_points = nx * ny * nz
        yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
        yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
        in_side = self.inside((xl3, yl3, zl3), (xr3, yr3, zr3))
        in_side = np.reshape(in_side[:,:,int(nz/2)], nx*ny)
        xl3 = np.reshape(xl3[:,:,int(nz/2)], nx*ny)
        yl3 = np.reshape(yl3[:,:,int(nz/2)], nx*ny)
        gd = np.where(in_side == 1)[0]
        ngd = np.size(gd)
        plt.plot(xl3[gd], yl3[gd], marker = '.', linestyle = '')
        plt.show(block = True)
        raise NotImplementedError("That's all for now!!!")
    
    def mapRho(self, xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, verbose = False):
        """
        Map density on a grid inside and around solid
        """
        if xmin == xmax:
            nx = 1
        if ymin == ymax:
            ny = 1
        if zmin == zmax:
            nz = 1
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        if nx > 1:
            dx = x[1] - x[0]
        else:
            dx = 0
        if ny > 1:
            dy = y[1] - y[0]
        else:
            dy = 0
        if nz > 1:
            dz = z[1] - z[0]
        else:
            dz = 0
            
        xl = x - 0.5 * dx
        xr = x + 0.5 * dx
        yl = y - 0.5 * dy
        yr = y + 0.5 * dy
        zl = z - 0.5 * dz
        zr = z + 0.5 * dz
        yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
        yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
        rho = self.vElementRho((xl3, yl3, zl3), (xr3, yr3, zr3), verbose = verbose, gridType = 'Cartesian')
        if nx == 1:
            if ny == 1:
                rho = np.reshape(rho, nz)
            elif nz == 1:
                rho = np.reshape(rho, ny)
            else:
                rho = np.reshape(rho, (ny, nz))
        else:
            if ny == 1 and nz == 1:
                pass #already in good shape
            elif ny == 1:
                rho = np.reshape(rho, (nx, nz))
            elif nz == 1:
                rho = np.reshape(rho, (nx, ny))
            else:
                pass
            
        return x, y, z, rho

    def infinitesimalVElementPotential(self, x, y, z, X, Y, Z, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, gridType = None):
        """Same thing as vElementPotential, but for dV->0 """
        rho = self.infinitesimalVElementRho(x, y, z, gridType = gridType)
        if self.kind == 'cylinder' and gridType == 'polar':
            #in this case, the grid is centered on (0,0,z0), so need to translate x and y to cbe centered on the cylinder
            x += self.x0
            y += self.y0
        d = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
        dUN = rho / d
        dUY = alpha * rho / d * np.exp(-d / lmbda)
        dU = int(getNewton) * dUN + int(getYukawa) * dUY
        return -dU

    def vElementPotential(self, xyzmin, xyzmax, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, verbose = False, gridType = None):
        """
        Compute potential created by infinitesimal volume element (xleft,yleft,zmin)-(xright,yright,zmax) at point (X,Y,Z) --WARNING! this is potential per volume, to accomodate simps integration
        """
        if gridType is None:
            raise TypeError("Please set gridType (this error message should be temporary, until all calls are cleaned up in the code... and a default value is provided)")
        rho = self.vElementRho(xyzmin, xyzmax, verbose = verbose, gridType = gridType)
        
        if self.kind == 'cylinder' and gridType == 'polar':
            #in this case, the grid is centered on (0,0,z0), so need to translate x and y to cbe centered on the cylinder
            rleft = xyzmin[0]
            thleft = xyzmin[1]
            rright = xyzmax[0]
            thright = xyzmax[1]
            xleft = rleft * np.cos(thleft) + self.x0
            yleft = rleft * np.sin(thleft) + self.y0
            xright = rright * np.cos(thright) + self.x0
            yright = rright * np.sin(thright) + self.y0
            zmin = xyzmin[2] #- self.z0
            zmax = xyzmax[2] #- self.z0
        else:
            xleft = xyzmin[0]
            yleft = xyzmin[1]
            zmin = xyzmin[2]
            xright = xyzmax[0]
            yright = xyzmax[1]
            zmax = xyzmax[2]
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        x = 0.5 * (xleft + xright)
        y = 0.5 * (yleft + yright)
        z = 0.5 * (zmin + zmax)
        d = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
        #dV = np.abs((xright - xleft) * (yright - yleft) * (zmax - zmin))
        #dU = rho * dV / d
        #dU = rho / d
        dUN = rho / d
        dUY = alpha * rho / d * np.exp(-d / lmbda)
        dU = int(getNewton) * dUN + int(getYukawa) * dUY
        return -dU

    def infinitesimalVElementAcceleration(self, x, y, z, X, Y, Z, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, gridType = None):
        """Same thing as vElementAcceleration, but for dV->0 """
        if gridType is None:
            raise TypeError("Please set gridType (this error message should be temporary, until all calls are cleaned up in the code... and a default value is provided)")
        rho = self.infinitesimalVElementRho(x, y, z, gridType = gridType)
        if self.kind == 'cylinder' and gridType == 'polar':
            #in this case, the grid is centered on (0,0,z0), so need to translate x and y to cbe centered on the cylinder
            x += self.x0
            y += self.y0
        d = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
        daN = -rho / d**3
        daxN = daN * (X-x)
        dayN = daN * (Y-y)
        dazN = daN * (Z-z)
        daY = -rho / d**3 * alpha * (1 + d / lmbda) * np.exp(-d / lmbda)
        daxY = daY * (X-x)
        dayY = daY * (Y-y)
        dazY = daY * (Z-z)
        dax = int(getNewton) * daxN + int(getYukawa) * daxY
        day = int(getNewton) * dayN + int(getYukawa) * dayY
        daz = int(getNewton) * dazN + int(getYukawa) * dazY
        return dax, day, daz

    def vElementAcceleration(self, xyzmin, xyzmax, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, verbose = False, gridType = None):
        """
        Compute acceleration created by infinitesimal volume element (xleft,yleft,zmin)-(xright,yright,zmax) at point (X,Y,Z) --WARNING! this is acceleration per volume, to accomodate simps integration
        """
        if gridType is None:
            raise TypeError("Please set gridType (this error message should be temporary, until all calls are cleaned up in the code... and a default value is provided)")
        # xleft = xyzmin[0]
        # yleft = xyzmin[1]
        # zmin = xyzmin[2]
        # xright = xyzmax[0]
        # yright = xyzmax[1]
        # zmax = xyzmax[2]
        # X = XYZ[0]
        # Y = XYZ[1]
        # Z = XYZ[2]
        
        rho = self.vElementRho(xyzmin, xyzmax, verbose = verbose, gridType = gridType)
        if self.kind == 'cylinder' and gridType == 'polar':
            #in this case, the grid is centered on (0,0,z0), so need to translate x and y to cbe centered on the cylinder
            rleft = xyzmin[0]
            thleft = xyzmin[1]
            rright = xyzmax[0]
            thright = xyzmax[1]
            xleft = rleft * np.cos(thleft) + self.x0
            yleft = rleft * np.sin(thleft) + self.y0
            xright = rright * np.cos(thright) + self.x0
            yright = rright * np.sin(thright) + self.y0
            zmin = xyzmin[2] #- self.z0
            zmax = xyzmax[2] #- self.z0
        else:
            xleft = xyzmin[0]
            yleft = xyzmin[1]
            zmin = xyzmin[2]
            xright = xyzmax[0]
            yright = xyzmax[1]
            zmax = xyzmax[2]
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        x = 0.5 * (xleft + xright)
        y = 0.5 * (yleft + yright)
        z = 0.5 * (zmin + zmax)
        d = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
        #da = -rho / d**3
        #dax = da * (X-x)
        #day = da * (Y-y)
        #daz = da * (Z-z)
        daN = -rho / d**3
        daxN = daN * (X-x)
        dayN = daN * (Y-y)
        dazN = daN * (Z-z)
        daY = -rho / d**3 * alpha * (1 + d / lmbda) * np.exp(-d / lmbda)
        daxY = daY * (X-x)
        dayY = daY * (Y-y)
        dazY = daY * (Z-z)
        dax = int(getNewton) * daxN + int(getYukawa) * daxY
        day = int(getNewton) * dayN + int(getYukawa) * dayY
        daz = int(getNewton) * dazN + int(getYukawa) * dazY
        zeros = np.where(d == 0)[0]
        dax[zeros] = 0
        day[zeros] = 0
        daz[zeros] = 0
        return dax, day, daz

        
    def infinitesimalVElementInertiaMoment(self, i, j, k, x, y, z):
        """Same as vElementInertiaMoment for dV->0"""
        rho = self.infinitesimalVElementRho(x, y, z)
        dI = rho * x**i * y**j * z**k #* dV
        return dI

    def vElementInertiaMoment(self, i, j, k, xyzmin, xyzmax, verbose = False):
        """
        Compute inertia moment contribution from infinitesimal volume element (xleft,yleft,zmin)-(xright,yright,zmax)
        """
        xleft = xyzmin[0]
        yleft = xyzmin[1]
        zmin = xyzmin[2]
        xright = xyzmax[0]
        yright = xyzmax[1]
        zmax = xyzmax[2]
        
        rho = self.vElementRho(xyzmin, xyzmax, verbose = verbose)
        x = 0.5 * (xleft + xright)
        y = 0.5 * (yleft + yright)
        z = 0.5 * (zmin + zmax)
        #dV = np.abs((xright - xleft) * (yright - yleft) * (zmax - zmin))
        dI = rho * x**i * y**j * z**k #* dV
        return dI

    def inertiaMoment(self, i, j, k, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, method = 'simps', mc_npoints = 1e6, memoryFriendly = False, verbose = True):
        """
        Compute ijk inertia moment (Int[rho*x^i*y^j*z^k dxdydz])
        method -- tplquad or simps
        This method only calls the numerical implementation, but is overloaded in some daughter classes.
        """
        return self.inertiaMoment_num(i, j, k, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, method = method, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
    
    def inertiaMoment_num(self, i, j, k, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, method = 'simps', mc_npoints = 1e6, memoryFriendly = False, verbose = True):
        """
        Compute ijk inertia moment (Int[rho*x^i*y^j*z^k dxdydz])
        method -- tplquad or simps
        """
        if method == 'simps':
            x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = self.mkGrid(dx_rel, dy_rel, dz_rel)
            #x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = cmpCubeElement(dx, dy, dz, self.radius, self.height)
            if verbose:
                print('inertia simps', i, j, k, nx, ny, nz)
        
            if memoryFriendly:
                dI = np.zeros((nx, ny, nz))
                for ix in range(nx):
                    if verbose and ix % 20 == 0:
                        print("inertiaMoment (" + str(i) + str(j) + str(k) + "): loop " + str(ix) + " / " + str(nx))
                    for iy in range(ny):
                        for iz in range(nz):
                            dI[ix, iy, iz] = self.vElementInertiaMoment(i, j, k, (xl[ix], yl[iy], zl[iz]), (xr[ix], yr[iy], zr[iz]))
            else:
                yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
                yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
                dI = self.vElementInertiaMoment(i, j, k, (xl3, yl3, zl3), (xr3, yr3, zr3))
            dIxy = simps(dI, x = z, axis = 2)
            dIx = simps(dIxy, x = y, axis = 1)
            I = simps(dIx, x = x, axis = 0)

        elif method == 'tplquad':
            gfun = lambda tx: self.tplquad_ymin_fun(tx) #lower boundary for y
            hfun = lambda tx: self.tplquad_ymax_fun(tx) #upper boundary for y
            qfun = lambda tx, ty: self.tplquad_zmin_fun(tx, ty) #lower boundary for z
            rfun = lambda tx, ty: self.tplquad_zmax_fun(tx, ty) #upper boundary for z

            def dI(zz, yy, xx):
                #order of z,y,x -> see tplquad help
                return self.infinitesimalVElementInertiaMoment(i, j, k, xx, yy, zz)
            print('inertia tplquad', i, j, k)
            I, err = tplquad(dI, self.tplquad_xmin, self.tplquad_xmax, gfun, hfun, qfun, rfun)
            if np.abs(err) >= 1e-3 * np.abs(I):
                print("WARNING! tplquad error is more than 1e-3 integral estimate...", I, err)

        elif method in ['mc', 'Monte Carlo']:
            #careful! This is for a cylinder only!!!
            raise NotImplementedError('Gotta find a new Monte Carlo integrator')
            # dI = lambda xyz: self.infinitesimalVElementInertiaMoment(i, j, k, xyz[0], xyz[1], xyz[2])
            # I, err = mcquad(dI, mc_npoints, [self.radius, self.radius, 0.5 * self.height], [-self.radius, -self.radius, -0.5 * self.height])
            # if np.abs(err) >= 1e-3 * np.abs(I):
            #     print("WARNING! mcquad error is more than 1e-3 integral estimate...", I, err)
        else:
            raise NotImplementedError('Method ' + method + ' not supported!')
            
        if i >= 10 or j >= 10 or k >= 10:
            raise NotImplementedError('i>=10, j>=10, k>10 not ready')
        else:
            iname = 'I' + str(i) + str(j) + str(k)
        setattr(self, iname, I)
        return I


    def cmpPotential_XYZ(self, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, verbose = True, timeit = False, gridType = 'Cartesian'):
        """
        Compute potential created by cylinder at position (X, Y, Z)
        method -- (regular grid or simps) to integrate on a regular grid, tplquad to use scipy tplquad, (mc or Monte Carlo) for a Monte Carlo integration or inertia moments to use inertia moments/Taylor expansion approximation
        im_method -- if method==inertia moments, method to use to compute inertia moments (only if they are not already set in the object): simps to integrate on a regular grid, tplquad to use scipy tplquad, (mc or Monte Carlo) for a Monte Carlo integration
        dx_rel, dy_rel, dz_rel -- if method==simps or regular grid, size of volume elements to use for integration, relative to overall object's size
        mc_npoints -- if method==mc or monte carlo, number of points to use for Monte Carlo integration
        memoryFriendly -- if method==simps or regular grid, use loops instead of huge arrays that may use all available memory
        pmax -- maximum order of decomposition (for method = 'ana multipolar')
        verbose -- write some info
        timeit -- measure computation time
        """
        if method in ['regular grid','simps']:
            return self.cmpPotential_XYZ_integreggrid(XYZ, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose, timeit = timeit, gridType = gridType)
        elif method == 'tplquad':
            return self.cmpPotential_XYZ_tplquad(XYZ, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, verbose = verbose, timeit = timeit)
        elif method in ['mc', 'Monte Carlo']:
            return self.cmpPotential_XYZ_mcquad(XYZ, npoints = mc_npoints, verbose = verbose, timeit = timeit)
        elif method == 'inertia moments':
            if getYukawa:
                raise NotImplementedError("How does Yukawa affect potential from inertia moments?")
            return self.cmpPotential_XYZ_inertiaMoments(XYZ, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, im_method = im_method, memoryFriendly = memoryFriendly, timeit = timeit )
        elif method == 'ana multipolar':
            return self.cmpPotential_ana_multipolar(XYZ, pmax = pmax)
        else:
            raise NotImplementedError('Method ' + method + ' not supported!')

    def cmpAcceleration_XYZ(self, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, verbose = True, timeit = False, gridType = 'Cartesian'):
        """
        Compute acceleration created by cylinder at position (X, Y, Z)
        method -- (regular grid or simps) to integrate on a regular grid, tplquad to use scipy tplquad, (mc or Monte Carlo) for a Monte Carlo integration or inertia moments to use inertia moments/Taylor expansion approximation
        im_method -- if method==inertia moments, method to use to compute inertia moments (only if they are not already set in the object): simps to integrate on a regular grid, tplquad to use scipy tplquad, (mc or Monte Carlo) for a Monte Carlo integration
        dx_rel, dy_rel, dz_rel -- if method==simps or regular grid, size of volume elements to use for integration, relative to overall object's size
        mc_npoints -- if method==mc or monte carlo, number of points to use for Monte Carlo integration
        memoryFriendly -- if method==simps or regular grid, use loops instead of huge arrays that may use all available memory
        pmax -- maximum order of decomposition (for method = 'ana multipolar')
        verbose -- write some info
        timeit -- measure computation time
        """
        if method in ['regular grid','simps']:
            return self.cmpAcceleration_XYZ_integreggrid(XYZ, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose, timeit = timeit, gridType = gridType)
        elif method == 'tplquad':
            return self.cmpAcceleration_XYZ_tplquad(XYZ, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, verbose = verbose, timeit = timeit)
        elif method == 'gradient':
            return self.cmpAcceleration_XYZ_grad(XYZ, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        elif method in ['mc', 'Monte Carlo']:
            raise NotImplementedError()
            return self.cmpAcceleration_XYZ_mcquad(XYZ, npoints = mc_npoints, verbose = verbose, timeit = timeit)
        elif method == 'inertia moments':
            return self.cmpAcceleration_XYZ_inertiaMoments(XYZ, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, im_method = im_method, memoryFriendly = memoryFriendly, timeit = timeit )
        elif method == 'ana multipolar':
            return self.cmpAcceleration_ana_multipolar(XYZ, pmax = pmax)
        else:
            raise NotImplementedError('Method ' + method + ' not supported!')


    def cmpPotential_XYZ_integreggrid(self, XYZ, dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, memoryFriendly = False, verbose = True, timeit = False, checkProp = True, gridType = 'Cartesian'):
        """
        XYZ -- (X, Y, Z)
        memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        if timeit:
            st = time.time()
        x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = self.mkGrid(dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, _type = gridType)
        #x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = cmpCubeElement(dx, dy, dz, self.radius, self.height)
        if verbose:
            print('grid integration', nx, ny, nz)

        #test proportion of grid really used for integration (proportion inside the solid)
        if checkProp:
            ng_points = nx * ny * nz
            yl3_cp, xl3_cp, zl3_cp = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
            yr3_cp, xr3_cp, zr3_cp = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
            in_side = self.inside((xl3_cp, yl3_cp, zl3_cp), (xr3_cp, yr3_cp, zr3_cp), gridType = gridType)
            n_inside = np.sum(in_side)
            prop = n_inside / ng_points * 100
            if prop != self.prevProp:
                self.prevProp = prop
                print("solid.cmpPotential_XYZ_integreggrid: " + str(prop) + "% of grid points are inside the solid")
            
        #all this block, to debug
        amr_work = False
        if amr_work:
            self.amrFunc(xl, xr, yl, yr, zl, zr, nx, ny, nz)
            
        if memoryFriendly:
            dU = np.zeros((nx, ny, nz))
            for ix in range(nx):
                if verbose and ix % 20 == 0:
                    print("cmpPotential_XYZ_integreggrid: loop " + str(ix) + " / " + str(nx))
                for iy in range(ny):
                    for iz in range(nz):
                        dU[ix, iy, iz] = self.vElementPotential((xl[ix], yl[iy], zl[iz]), (xr[ix], yr[iy], zr[iz]), (X, Y, Z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, gridType = gridType)
        else:
            yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
            yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
            dU = self.vElementPotential((xl3, yl3, zl3), (xr3, yr3, zr3), (X, Y, Z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, gridType = gridType)
        dUxy = simps(dU, x = z, axis = 2)
        if gridType == 'Cartesian':
            dUx = simps(dUxy, x = y, axis = 1)
            U = simps(dUx, x = x, axis = 0)
        elif gridType == 'polar':
            #only for cylinder (should be!), so it's okay to have integration on out of this condition (above)
            dUx = simps(dUxy, x = y, axis = 1)
            U = simps(x * dUx, x = x, axis = 0)
        else:
            raise ValueError('?!?')
        U *= G
        if timeit:
            et = time.time()
            return (U, et - st)
        return U
        
    def cmpPotential_XYZ_tplquad(self, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, verbose = True, timeit = False):
        """
        XYZ -- (X, Y, Z)
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        if timeit:
            st = time.time()
        #dU = lambda zz, yy, xx: self.infinitesimalVElementPotential(xx, yy, zz, X, Y, Z) #order of z,y,x -> see tplquad help
        gfun = lambda tx: self.tplquad_ymin_fun(tx) #lower boundary for y
        hfun = lambda tx: self.tplquad_ymax_fun(tx) #upper boundary for y
        qfun = lambda tx, ty: self.tplquad_zmin_fun(tx, ty) #lower boundary for z
        rfun = lambda tx, ty: self.tplquad_zmax_fun(tx, ty) #upper boundary for z

        def dU(zz, yy, xx):
            #order of z,y,x -> see tplquad help
            return self.infinitesimalVElementPotential(xx, yy, zz, X, Y, Z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
        
        U, err = tplquad(dU, self.tplquad_xmin, self.tplquad_xmax, gfun, hfun, qfun, rfun)
        if np.abs(err) >= 1e-3 * np.abs(U):
            print("WARNING! tplquad error is more than 1e-3 integral estimate...", U, err)
        U *= G
        if timeit:
            et = time.time()
            return (U, et - st)
        return U

    def cmpPotential_XYZ_mcquad(self, XYZ, npoints = 1e6, verbose = True, timeit = False):
        raise NotImplementedError("Must be implemented in daughter classes???")

    #def cmpPotential_XYZ_inertiaMoments(self, XYZ, dx = None, dy = None, dz = None, im_method = 'simps', memoryFriendly = False, timeit = False):
    #    raise NotImplementedError("Must be implemented in daughter classes???")

    def cmpPotential_XYZ_inertiaMoments(self, XYZ, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, im_method = 'simps', memoryFriendly = False, timeit = False):
        """
        see MIC-NT-SY-0-6002-OCA
        XYZ -- (X, Y, Z)
        memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        if timeit:
            st = time.time()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if i + j + k == 0:
                        continue
                    if i + j + k > 3:
                        continue
                    ilocal = getattr(self, 'I' + str(i) + str(j) + str(k))
                    if ilocal is None:
                        self.inertiaMoment(i, j, k, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, method = im_method, memoryFriendly = memoryFriendly)
        mass = self.cmpMass()
        R = np.sqrt(X**2 + Y**2 + Z**2)
        V0 = mass / R
        V3 = (X * self.I100 + Y * self.I010 + Z * self.I001) / R**3
        V5 = (3 * (X * Y * self.I110 + X * Z * self.I101 + Y * Z * self.I011) + 0.5 * ((3. * X**2 - R**2) * self.I200 + (3. * Y**2 - R**2) * self.I020 + (3. * Z**2 - R**2) * self.I002)) / R**5
        V7 = ((15. * X**2 - 3. * R**2) * (Y * self.I210 + Z * self.I201) + (15. * Y**2 - 3. * R**2) * (X * self.I120 + Z * self.I021) + (15. * Z**2 - 3. * R**2) * (X * self.I102 + Y * self.I012) + 30. * X * Y * Z * self.I111 + (5. * X**2 - 3. * R**2) * X * self.I300 + (5. * Y**2 - 3. * R**2) * Y * self.I030 + (5. * Z**2 - 3. * R**2) * Z * self.I003 ) / (2. * R**7)
        V = G * (V0 + V3 + V5 + V7)
        #V = G * V0 #Gm/R
        #V = G * mass * (V0 + V3 + V7)
        if timeit:
            et = time.time()
            return (V, et - st)
        return V

    def cmpPotential_ana_multipolar(self, XYZ, pmax = 15):
        """
        Lockerbie et al 1993
        Only Newtonian! and cylinder!
        """
        raise NotImplementedError("Only for cylinder and Newtonian gravity!")

    def cmpAcceleration_ana_multipolar(self, XYZ, pmax = 15):
        """
        Lockerbie et al 1993
        Only Newtonian! and cylinder!
        """
        raise NotImplementedError("Only for cylinder and Newtonian gravity!")

    def cmpAcceleration_XYZ_integreggrid(self, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = True, timeit = False, checkProp = True, gridType = 'Cartesian'):
        """
        XYZ -- (X, Y, Z)
        memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        if timeit:
            st = time.time()
        x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = self.mkGrid(dx_rel, dy_rel, dz_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, _type = gridType)
        #x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = cmpCubeElement(dx, dy, dz, self.radius, self.height)
        if verbose:
            print('grid integration', nx, ny, nz)

        #test proportion of grid really used for integration (proportion inside the solid)
        if checkProp:
            ng_points = nx * ny * nz
            yl3_cp, xl3_cp, zl3_cp = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
            yr3_cp, xr3_cp, zr3_cp = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
            in_side = self.inside((xl3_cp, yl3_cp, zl3_cp), (xr3_cp, yr3_cp, zr3_cp), gridType = gridType)
            n_inside = np.sum(in_side)
            prop = n_inside / ng_points * 100
            if prop != self.prevProp:
                self.prevProp = prop
                print("solid.cmpAcceleration_XYZ_integreggrid: " + str(prop) + "% of grid points are inside the solid")
            
        if memoryFriendly:
            dax = np.zeros((nx, ny, nz))
            day = np.zeros((nx, ny, nz))
            daz = np.zeros((nx, ny, nz))
            for ix in range(nx):
                if verbose and ix % 20 == 0:
                    print("cmpAcceleration_XYZ_integreggrid: loop " + str(ix) + " / " + str(nx))
                for iy in range(ny):
                    for iz in range(nz):
                        dax[ix, iy, iz], day[ix, iy, iz], daz[ix, iy, iz] = self.vElementAcceleration((xl[ix], yl[iy], zl[iz]), (xr[ix], yr[iy], zr[iz]), (X, Y, Z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, gridType = gridType)
        else:
            yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
            yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
            dax, day, daz = self.vElementAcceleration((xl3, yl3, zl3), (xr3, yr3, zr3), (X, Y, Z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, gridType = gridType)
        dax_xy = simps(dax, x = z, axis = 2)
        if gridType == 'Cartesian':
            dax_x = simps(dax_xy, x = y, axis = 1)
            ax = simps(dax_x, x = x, axis = 0)
        elif gridType == 'polar':
            #only for cylinder (should be!), so it's okay to have integration on out of this condition (above)
            dax_x = simps(dax_xy, x = y, axis = 1)
            ax = simps(x * dax_x, x = x, axis = 0)
        else:
            raise ValueError('?!?')
        ax *= G
        day_xy = simps(day, x = z, axis = 2)
        if gridType == 'Cartesian':
            day_x = simps(day_xy, x = y, axis = 1)
            ay = simps(day_x, x = x, axis = 0)
        elif gridType == 'polar':
            #only for cylinder (should be!), so it's okay to have integration on out of this condition (above)
            day_x = simps(day_xy, x = y, axis = 1)
            ay = simps(x * day_x, x = x, axis = 0)
        else:
            raise ValueError('?!?')
        ay *= G
        daz_xy = simps(daz, x = z, axis = 2)
        if gridType == 'Cartesian':
            daz_x = simps(daz_xy, x = y, axis = 1)
            az = simps(daz_x, x = x, axis = 0)
        elif gridType == 'polar':
            #only for cylinder (should be!), so it's okay to have integration on out of this condition (above)
            daz_x = simps(daz_xy, x = y, axis = 1)
            az = simps(x * daz_x, x = x, axis = 0)
        else:
            raise ValueError('?!?')
        az *= G

        if timeit:
            et = time.time()
            return (ax, ay, az, et - st)
        return ax, ay, az

    def cmpAcceleration_XYZ_tplquad(self, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, verbose = True, timeit = False):
        """
        XYZ -- (X, Y, Z)
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        if timeit:
            st = time.time()
        #dU = lambda zz, yy, xx: self.infinitesimalVElementPotential(xx, yy, zz, X, Y, Z) #order of z,y,x -> see tplquad help
        gfun = lambda tx: self.tplquad_ymin_fun(tx) #lower boundary for y
        hfun = lambda tx: self.tplquad_ymax_fun(tx) #upper boundary for y
        qfun = lambda tx, ty: self.tplquad_zmin_fun(tx, ty) #lower boundary for z
        rfun = lambda tx, ty: self.tplquad_zmax_fun(tx, ty) #upper boundary for z

        def dax(zz, yy, xx):
            #order of z,y,x -> see tplquad help
            return self.infinitesimalVElementAcceleration(xx, yy, zz, X, Y, Z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)[0]

        def day(zz, yy, xx):
            #order of z,y,x -> see tplquad help
            return self.infinitesimalVElementAcceleration(xx, yy, zz, X, Y, Z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)[1]

        def daz(zz, yy, xx):
            #order of z,y,x -> see tplquad help
            return self.infinitesimalVElementAcceleration(xx, yy, zz, X, Y, Z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)[2]
        
        ax, err = tplquad(dax, self.tplquad_xmin, self.tplquad_xmax, gfun, hfun, qfun, rfun)
        if np.abs(err) >= 1e-3 * np.abs(ax):
            print("WARNING! tplquad error is more than 1e-3 integral estimate...", ax, err)
        ax *= G
        ay, err = tplquad(day, self.tplquad_xmin, self.tplquad_xmax, gfun, hfun, qfun, rfun)
        if np.abs(err) >= 1e-3 * np.abs(ay):
            print("WARNING! tplquad error is more than 1e-3 integral estimate...", ay, err)
        ay *= G
        az, err = tplquad(daz, self.tplquad_xmin, self.tplquad_xmax, gfun, hfun, qfun, rfun)
        if np.abs(err) >= 1e-3 * np.abs(az):
            print("WARNING! tplquad error is more than 1e-3 integral estimate...", az, err)
        az *= G
        
        if timeit:
            et = time.time()
            return (ax, ay, az, et - st)
        return ax, ay, az

    def cmpAcceleration_XYZ_grad(self, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = True):
        """
        XYZ -- (X, Y, Z)
        memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]

        vfunc = lambda p: self.cmpPotential_XYZ((p[0], p[1], p[2]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        dvfunc = nd.Gradient(vfunc)
        dv = dvfunc([X, Y, Z])
        ax = dv[0]
        ay = dv[1]
        az = dv[2]
        return ax, ay, az

    def cmpAcceleration_XYZ_inertiaMoments(self, XYZ, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, im_method = 'simps', memoryFriendly = False, timeit = False, message_period = 30):
        """
        see MIC-NT-SY-0-6002-OCA.
        XYZ -- (X, Y, Z)
        memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        if timeit:
            st = time.time()
        nsteps = 4**3 #4 because we go up to 3rd moments of inertia
        nth_step = 0
        j_message = 0
        last_message_time = time.time()
        time_start = time.time()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    nth_step += 1
                    if i + j + k == 0:
                        #I000 not used... this is just the mass of the object!
                        continue
                    if i + j + k > 3:
                        continue
                    ilocal = getattr(self, 'I' + str(i) + str(j) + str(k))
                    if ilocal is None:
                        j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "cylinder.cmpAcceleration_XYZ_inertiaMoments")
                        self.inertiaMoment(i, j, k, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, method = im_method, memoryFriendly = memoryFriendly, verbose = False)
        mass = self.cmpMass()
        R = np.sqrt(X**2 + Y**2 + Z**2)

        kx3 = X / R**3
        kx5 = ((3 * X**2 - R**2) * self.I100 + 3 * X * (Y * self.I010 + Z * self.I001)) / R**5 / mass
        kx7 = ( (5 * X**2 - R**2) * (Y * self.I110 + Z * self.I101) + 5 * X * Y * Z * self.I011 + X / 2 * ( (5 * X**2 - 3 * R**2) * self.I200 + (5 * Y**2 - R**2) * self.I020 + (5 * Z**2 - R**2) * self.I002 ) ) * 3. / R**7 / mass
        kx9 = ( 15 * X * ( (7 * X**2 - 3 * R**2) * (Y * self.I210 + Z * self.I201) + (7 * Y**2 - R**2) * Z * self.I021 + (7 * Z**2 - R**2) * Y * self.I012 ) + 30 * Y * Z * (7 * X**2 - R**2) * self.I111 + 3 * ( (35 * X**2 * Y**2 - 5 * X**2 * R**2 - 5 * Y**2 * R**2 + R**4) * self.I120 + (35 * X**2 * Z**2 - 5 * X**2 * R**2 - 5 * Z**2 * R**2 + R**4) * self.I102 ) + (35 * X**4 - 30 * X**2 * R**2 + 3 * R**4) * self.I300 + 5 * X * Y * (7 * Y**2 - 3 * R**2) * self.I030 + 5 * X * Z * (7 * Z**2 - 3 * R**2) * self.I003 ) / (2 * R**9) / mass

        ky3 = Y / R**3
        ky5 = ((3 * Y**2 - R**2) * self.I010 + 3 * Y * (Z * self.I001 + X * self.I100)) / R**5 / mass
        ky7 = ( (5 * Y**2 - R**2) * (Z * self.I011 + X * self.I110) + 5 * X * Y * Z * self.I101 + Y / 2 * ( (5 * Y**2 - 3 * R**2) * self.I020 + (5 * Z**2 - R**2) * self.I002 + (5 * X**2 - R**2) * self.I200 ) ) * 3. / R**7 / mass
        ky9 = ( 15 * Y * ( (7 * Y**2 - 3 * R**2) * (Z * self.I021 + X * self.I120) + (7 * Z**2 - R**2) * X * self.I102 + (7 * X**2 - R**2) * Z * self.I201 ) + 30 * X * Z * (7 * Y**2 - R**2) * self.I111 + 3 * ( (35 * Y**2 * Z**2 - 5 * Y**2 * R**2 - 5 * Z**2 * R**2 + R**4) * self.I012 + (35 * Y**2 * X**2 - 5 * Y**2 * R**2 - 5 * X**2 * R**2 + R**4) * self.I210 ) + (35 * Y**4 - 30 * Y**2 * R**2 + 3 * R**4) * self.I030 + 5 * Y * Z * (7 * Z**2 - 3 * R**2) * self.I003 + 5 * X * Y * (7 * X**2 - 3 * R**2) * self.I300 ) / (2 * R**9) / mass

        kz3 = Z / R**3
        kz5 = ((3 * Z**2 - R**2) * self.I001 + 3 * Z * (X * self.I100 + Y * self.I010)) / R**5 / mass
        kz7 = ( (5 * Z**2 - R**2) * (X * self.I101 + Y * self.I011) + 5 * X * Y * Z * self.I110 + Z / 2 * ( (5 * Z**2 - 3 * R**2) * self.I002 + (5 * X**2 - R**2) * self.I200 + (5 * Y**2 - R**2) * self.I020 ) ) * 3. / R**7 / mass
        kz9 = ( 15 * Z * ( (7 * Z**2 - 3 * R**2) * (X * self.I102 + Y * self.I012) + (7 * X**2 - R**2) * Y * self.I210 + (7 * Z**2 - R**2) * X * self.I120 ) + 30 * X * Y * (7 * Z**2 - R**2) * self.I111 + 3 * ( (35 * X**2 * Z**2 - 5 * Z**2 * R**2 - 5 * X**2 * R**2 + R**4) * self.I201 + (35 * Z**2 * Y**2 - 5 * Z**2 * R**2 - 5 * Y**2 * R**2 + R**4) * self.I021 ) + (35 * Z**4 - 30 * Z**2 * R**2 + 3 * R**4) * self.I003 + 5 * X * Z * (7 * X**2 - 3 * R**2) * self.I300 + 5 * Y * Z * (7 * Y**2 - 3 * R**2) * self.I030 ) / (2 * R**9) / mass

        #- b/c MIC-NT-SY-0-6002-OCA gives acceleration ue to point mass on cylinder (the opposite of what we want here)
        ax = -G * mass * (kx3 + kx5 + kx7 + kx9)
        ay = -G * mass * (ky3 + ky5 + ky7 + ky9)
        az = -G * mass * (kz3 + kz5 + kz7 + kz9)
        #ax = -G * mass * (kx3 + kx5 + kx9)
        #ay = -G * mass * (ky3 + ky5 + ky9)
        #az = -G * mass * (kz3 + kz5 + kz9)
        #V = G * V0 #Gm/R
        if timeit:
            et = time.time()
            return (ax, ay, az, et - st)
        #print(ax, ay, az)
        return ax, ay, az
    
    def cmpGGT_XYZ(self, XYZ, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = True):
        """
        XYZ -- (X, Y, Z)
        memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]

        vfunc = lambda p: self.cmpPotential_XYZ((p[0], p[1], p[2]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        ddvfunc = nd.Hessian(vfunc)
        ddv = ddvfunc([X, Y, Z])
        Txx = ddv[0,0]
        Txy = ddv[0,1]
        Txz = ddv[0,2]
        Tyy = ddv[1,1]
        Tyz = ddv[1,2]
        Tzz = ddv[2,2]
        return Txx, Txy, Txz, Tyy, Tyz, Tzz

        
    
    def v_grid(self, x = 0, y = 0, z = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, verbose = False, message_period = 60, gridType = 'Cartesian'):
        """compute potential on a 1D, 2D or 3D grid"""
        nx = np.size(x)
        ny = np.size(y)
        nz = np.size(z)
        if nx == 1 and ny == 1 and nz == 1:
            if x == 0 and y == 0 and z == 0:
                raise ValueError("At least one of x, y or z must be set")
            return x, y, z, self.cmpPotential_XYZ((x, y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)
        #return x, y, z, self.cmpPotential_XYZ_integreggrid((x, y, z), dx = dx, dy = dy, dz = dz)

        locx = scalar2array(x)
        locy = scalar2array(y)
        locz = scalar2array(z)
        nsteps = nx * ny * nz
        time_start = time.time()
        j_message = 0
        nth_step = 0
        last_message_time = time.time()
        if nx > 1 and ny == 1 and nz == 1:
            v = np.zeros(nx)
            for i in range(nx):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.v_grid")
                #if i % 2 == 0:
                #    print("solid.v_grid: computing step " + str(i+1) + " / " + str(nx))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((locx[i], y, z), dx = dx, dy = dy, dz = dz)
                v[i] = self.cmpPotential_XYZ((locx[i], y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)
        elif nx == 1 and ny > 1 and nz == 1:
            v = np.zeros(ny)
            for i in range(ny):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.v_grid")
                #if i % 2 == 0:
                #    print("solid.v_grid: computing step " + str(i+1) + " / " + str(ny))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((x, locy[i], z), dx = dx, dy = dy, dz = dz)
                v[i] = self.cmpPotential_XYZ((x, locy[i], z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)
        elif nx == 1 and ny == 1 and nz > 1:
            v = np.zeros(nz)
            for i in range(nz):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.v_grid")
                #if i % 2 == 0:
                #    print("solid.v_grid: computing step " + str(i+1) + " / " + str(nz))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((x, y, locz[i]), dx = dx, dy = dy, dz = dz)
                v[i] = self.cmpPotential_XYZ((x, y, locz[i]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)
        else:
            v = np.zeros((nx, ny, nz))
            for i in range(nx):
                #if nx > 1:
                #    if i % 2 == 0:
                #        print("solid.v_grid: computing step " + str(i+1) + " / " + str(nx))
                for j in range(ny):
                    #if nx == 1 and j % 50 == 0:
                    #    print("solid.v_grid: computing step " + str(j+1) + " / " + str(ny))
                    for k in range(nz):
                        nth_step += 1
                        j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.v_grid")
                        #v[i,j,k] = self.cmpPotential_XYZ_integreggrid((locx[i], locy[j], locz[k]), dx = dx, dy = dy, dz = dz)
                        v[i,j,k] = self.cmpPotential_XYZ((locx[i], locy[j], locz[k]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)

        if nx == 1:
            if ny == 1:
                v = np.reshape(v, nz)
            elif nz == 1:
                v = np.reshape(v, ny)
            else:
                v = np.reshape(v, (ny, nz))
        else:
            if ny == 1 and nz == 1:
                pass #already in good shape
            elif ny == 1:
                v = np.reshape(v, (nx, nz))
            elif nz == 1:
                v = np.reshape(v, (nx, ny))
            else:
                pass
        return x, y, z, v

    
    def a_grid(self, x = 0, y = 0, z = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, verbose = False, message_period = 60, gridType = 'Cartesian'):
        """compute potential on a 1D, 2D or 3D grid"""
        nx = np.size(x)
        ny = np.size(y)
        nz = np.size(z)
        if nx == 1 and ny == 1 and nz == 1:
            if x == 0 and y == 0 and z == 0:
                raise ValueError("At least one of x, y or z must be set")
            return x, y, z, self.cmpAcceleration_XYZ((x, y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, pmax = pmax, timeit = False, gridType = gridType)
        #return x, y, z, self.cmpPotential_XYZ_integreggrid((x, y, z), dx = dx, dy = dy, dz = dz)

        locx = scalar2array(x)
        locy = scalar2array(y)
        locz = scalar2array(z)
        nsteps = nx * ny * nz
        time_start = time.time()
        j_message = 0
        nth_step = 0
        last_message_time = time.time()
        if nx > 1 and ny == 1 and nz == 1:
            ax = np.zeros(nx)
            ay = np.zeros(nx)
            az = np.zeros(nx)
            for i in range(nx):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.a_grid")
                    #v[i] = self.cmpPotential_XYZ_integreggrid((locx[i], y, z), dx = dx, dy = dy, dz = dz)
                ax[i], ay[i], az[i] = self.cmpAcceleration_XYZ((locx[i], y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)
        elif nx == 1 and ny > 1 and nz == 1:
            ax = np.zeros(ny)
            ay = np.zeros(ny)
            az = np.zeros(ny)
            for i in range(ny):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.a_grid")
                #if i % 2 == 0:
                #    print("solid.a_grid: computing step " + str(i+1) + " / " + str(ny))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((x, locy[i], z), dx = dx, dy = dy, dz = dz)
                ax[i], ay[i], az[i] = self.cmpAcceleration_XYZ((x, locy[i], z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)
        elif nx == 1 and ny == 1 and nz > 1:
            ax = np.zeros(nz)
            ay = np.zeros(nz)
            az = np.zeros(nz)
            for i in range(nz):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.a_grid")
                #if i % 2 == 0:
                #    print("solid.a_grid: computing step " + str(i+1) + " / " + str(nz))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((x, y, locz[i]), dx = dx, dy = dy, dz = dz)
                ax[i], ay[i], az[i] = self.cmpAcceleration_XYZ((x, y, locz[i]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)
        else:
            ax = np.zeros((nx, ny, nz))
            ay = np.zeros((nx, ny, nz))
            az = np.zeros((nx, ny, nz))
            for i in range(nx):
                #if nx > 1:
                #    if i % 2 == 0:
                #        print("solid.a_grid: computing step " + str(i+1) + " / " + str(nx))
                for j in range(ny):
                    #if nx == 1 and j % 50 == 0:
                    #    print("solid.a_grid: computing step " + str(j+1) + " / " + str(ny))
                    for k in range(nz):
                        nth_step += 1
                        j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.a_grid")
                        #v[i,j,k] = self.cmpPotential_XYZ_integreggrid((locx[i], locy[j], locz[k]), dx = dx, dy = dy, dz = dz)
                        ax[i,j,k], ay[i,j,k], az[i,j,k] = self.cmpAcceleration_XYZ((locx[i], locy[j], locz[k]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, timeit = False, gridType = gridType)

        if nx == 1:
            if ny == 1:
                ax = np.reshape(ax, nz)
                ay = np.reshape(ay, nz)
                az = np.reshape(az, nz)
            elif nz == 1:
                ax = np.reshape(ax, ny)
                ay = np.reshape(ay, ny)
                az = np.reshape(az, ny)
            else:
                ax = np.reshape(ax, (ny, nz))
                ay = np.reshape(ay, (ny, nz))
                az = np.reshape(az, (ny, nz))
        else:
            if ny == 1 and nz == 1:
                pass #already in good shape
            elif ny == 1:
                ax = np.reshape(ax, (nx, nz))
                ay = np.reshape(ay, (nx, nz))
                az = np.reshape(az, (nx, nz))
            elif nz == 1:
                ax = np.reshape(ax, (nx, ny))
                ay = np.reshape(ay, (nx, ny))
                az = np.reshape(az, (nx, ny))
            else:
                pass
        return x, y, z, ax, ay, az

    def a_grid2(self, xyz = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, verbose = False, message_period = 60, gridType = 'Cartesian'):
        """
        same thing as a_grid, but with x,y,z correlated (irregular grid)
        xyz -- [N, 3] array containing triplets of x,y,z of the irregular grid
        """
        if np.size(np.shape(xyz)) != 2:
            raise TypeError("Bad xyz. Must be a [N, 3] array containing triplets of x,y,z of the irregular grid")
        if np.shape(xyz)[1] != 3:
            raise TypeError("Bad xyz. Must be a [N, 3] array containing triplets of x,y,z of the irregular grid")
        N = np.shape(xyz)[0]
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]
        nsteps = N
        ax = np.zeros(N)
        ay = np.zeros(N)
        az = np.zeros(N)
        time_start = time.time()
        j_message = 0
        nth_step = 0
        last_message_time = time.time()
        for i in range(N):
            nth_step += 1
            j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.a_grid2")
            ax[i], ay[i], az[i] = self.cmpAcceleration_XYZ((x[i], y[i], z[i]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, pmax = pmax, timeit = False, gridType = gridType)
        return x, y, z, ax, ay, az


    def T_grid(self, x = 0, y = 0, z = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False, message_period = 60):
        """compute potential on a 1D, 2D or 3D grid"""
        nx = np.size(x)
        ny = np.size(y)
        nz = np.size(z)
        if nx == 1 and ny == 1 and nz == 1:
            if x == 0 and y == 0 and z == 0:
                raise ValueError("At least one of x, y or z must be set")
            return x, y, z, self.cmpGGT_XYZ((x, y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        #return x, y, z, self.cmpPotential_XYZ_integreggrid((x, y, z), dx = dx, dy = dy, dz = dz)

        locx = scalar2array(x)
        locy = scalar2array(y)
        locz = scalar2array(z)
        nsteps = nx * ny * nz
        time_start = time.time()
        j_message = 0
        nth_step = 0
        last_message_time = time.time()
        if nx > 1 and ny == 1 and nz == 1:
            Txx = np.zeros(nx)
            Txy = np.zeros(nx)
            Txz = np.zeros(nx)
            Tyy = np.zeros(nx)
            Tyz = np.zeros(nx)
            Tzz = np.zeros(nx)
            for i in range(nx):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.T_grid")
                #if i % 2 == 0:
                #    print("solid.T_grid: computing step " + str(i+1) + " / " + str(nx))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((locx[i], y, z), dx = dx, dy = dy, dz = dz)
                Txx[i], Txy[i], Txz[i], Tyy[i], Tyz[i], Tzz[i] = self.cmpGGT_XYZ((locx[i], y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        elif nx == 1 and ny > 1 and nz == 1:
            Txx = np.zeros(ny)
            Txy = np.zeros(ny)
            Txz = np.zeros(ny)
            Tyy = np.zeros(ny)
            Tyz = np.zeros(ny)
            Tzz = np.zeros(ny)
            for i in range(ny):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.T_grid")
                #if i % 2 == 0:
                #    print("solid.T_grid: computing step " + str(i+1) + " / " + str(ny))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((x, locy[i], z), dx = dx, dy = dy, dz = dz)
                Txx[i], Txy[i], Txz[i], Tyy[i], Tyz[i], Tzz[i] = self.cmpGGT_XYZ((x, locy[i], z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        elif nx == 1 and ny == 1 and nz > 1:
            Txx = np.zeros(nz)
            Txy = np.zeros(nz)
            Txz = np.zeros(nz)
            Tyy = np.zeros(nz)
            Tyz = np.zeros(nz)
            Tzz = np.zeros(nz)
            for i in range(nz):
                nth_step += 1
                j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.T_grid")
                #if i % 2 == 0:
                #    print("solid.T_grid: computing step " + str(i+1) + " / " + str(nz))
                    #v[i] = self.cmpPotential_XYZ_integreggrid((x, y, locz[i]), dx = dx, dy = dy, dz = dz)
                Txx[i], Txy[i], Txz[i], Tyy[i], Tyz[i], Tzz[i] = self.cmpGGT_XYZ((x, y, locz[i]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        else:
            Txx = np.zeros((nx, ny, nz))
            Txy = np.zeros((nx, ny, nz))
            Txz = np.zeros((nx, ny, nz))
            Tyy = np.zeros((nx, ny, nz))
            Tyz = np.zeros((nx, ny, nz))
            Tzz = np.zeros((nx, ny, nz))
            for i in range(nx):
                #if nx > 1:
                #    if i % 2 == 0:
                #        print("solid.T_grid: computing step " + str(i+1) + " / " + str(nx))
                for j in range(ny):
                    #if nx == 1 and j % 50 == 0:
                    #    print("solid.T_grid: computing step " + str(j+1) + " / " + str(ny))
                    for k in range(nz):
                        nth_step += 1
                        j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = message_period, origin = "solid.T_grid")
                        #v[i,j,k] = self.cmpPotential_XYZ_integreggrid((locx[i], locy[j], locz[k]), dx = dx, dy = dy, dz = dz)
                        Txx[i,j,k], Txy[i,j,k], Txz[i,j,k], Tyy[i,j,k], Tyz[i,j,k], Tzz[i,j,k] = self.cmpGGT_XYZ((locx[i], locy[j], locz[k]), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)

        if nx == 1:
            if ny == 1:
                Txx = np.reshape(Txx, nz)
                Txy = np.reshape(Txy, nz)
                Txz = np.reshape(Txz, nz)
                Tyy = np.reshape(Tyy, nz)
                Tyz = np.reshape(Tyz, nz)
                Tzz = np.reshape(Tzz, nz)
            elif nz == 1:
                Txx = np.reshape(Txx, ny)
                Txy = np.reshape(Txy, ny)
                Txz = np.reshape(Txz, ny)
                Tyy = np.reshape(Tyy, ny)
                Tyz = np.reshape(Tyz, ny)
                Tzz = np.reshape(Tzz, ny)
            else:
                Txx = np.reshape(Txx, (ny, nz))
                Txy = np.reshape(Txy, (ny, nz))
                Txz = np.reshape(Txz, (ny, nz))
                Tyy = np.reshape(Tyy, (ny, nz))
                Tyz = np.reshape(Tyz, (ny, nz))
                Tzz = np.reshape(Tzz, (ny, nz))
        else:
            if ny == 1 and nz == 1:
                pass #already in good shape
            elif ny == 1:
                Txx = np.reshape(Txx, (nx, nz))
                Txy = np.reshape(Txy, (nx, nz))
                Txz = np.reshape(Txz, (nx, nz))
                Tyy = np.reshape(Tyy, (nx, nz))
                Tyz = np.reshape(Tyz, (nx, nz))
                Tzz = np.reshape(Tzz, (nx, nz))
            elif nz == 1:
                Txx = np.reshape(Txx, (nx, ny))
                Txy = np.reshape(Txy, (nx, ny))
                Txz = np.reshape(Txz, (nx, ny))
                Tyy = np.reshape(Tyy, (nx, ny))
                Tyz = np.reshape(Tyz, (nx, ny))
                Tzz = np.reshape(Tzz, (nx, ny))
            else:
                pass
        return x, y, z, Txx, Txy, Txz, Tyy, Tyz, Tzz
    

    def plt_vAlongAxis(self, axis, _min, _max, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, pltAxis = None, objectRef = True, n = 100, orthogonalPlaneCross = (0.,0.), log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, hold = False):
        return self.plt_vatAlongAxis('potential', axis, _min, _max, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, pltAxis = pltAxis, objectRef = objectRef, n = n, orthogonalPlaneCross = orthogonalPlaneCross, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = hold)

    def plt_aAlongAxis(self, axis, _min, _max, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, acomp = None, pltAxis = None, objectRef = True, n = 100, orthogonalPlaneCross = (0.,0.), log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, hold = False, ax = None, ay = None, az = None):
        return self.plt_vatAlongAxis('acceleration', axis, _min, _max, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = acomp, pltAxis = pltAxis, objectRef = objectRef, n = n, orthogonalPlaneCross = orthogonalPlaneCross, log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = hold, ax = ax, ay = ay, az = az)

    def plt_TAlongAxis(self, axis, _min, _max, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, Tcomp = None, pltAxis = None, objectRef = True, n = 100, orthogonalPlaneCross = (0.,0.), log = False, yscale = 'linear', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, hold = False, Txx = None, Txy = None, Txz = None, Tyy = None, Tyz = None, Tzz = None):
        return self.plt_vatAlongAxis('GGT', axis, _min, _max, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, Tcomp = Tcomp, pltAxis = pltAxis, objectRef = objectRef, n = n, orthogonalPlaneCross = orthogonalPlaneCross, log = log, yscale = yscale, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = hold, Txx = Txx, Txy = Txy, Txz = Txz, Tyy = Tyy, Tyz = Tyz, Tzz = Tzz)
    
    def plt_vatAlongAxis(self, _type, axis, _min, _max, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, acomp = None, Tcomp = None, pltAxis = None, objectRef = True, n = 100, orthogonalPlaneCross = (0.,0.), log = False, yscale = 'linear', method = 'tplquad', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, hold = False, ax = None, ay = None, az = None, Txx = None, Txy = None, Txz = None, Tyy = None, Tyz = None, Tzz = None):
        """
        Plot potential as a function of height (parallel to axis)
        _min, _max - limits on axis to plot [m]
        pltAxis -- matplotlib axis
        acomp -- if _type is acceleration, component to plot
        ax, ay, az -- acceleration components corresponding to local coordinates, if already computed (to avoid computing them again)
        objectRef -- if set, center plot on object origin (not necessarily its center!), else use glocal coordinates
        orthogonalPlaneCross -- coordinates of other axes, where axis crosses orthogonal plane (e.g., if plotting potential along z, (x0,y0))
        """
        if _max <= _min:
            raise ValueError("_min must be < _max")
        if not log:
            a = np.linspace(_min, _max, n)
        else:
            if _min < 0:
                raise ValueError("For log, _min must be strictly >0")
            a = np.logspace(np.log10(_min), np.log10(_max), n)
            
        if axis == 'z':
            axis2_name = 'x'
            axis3_name = 'y'
            x = orthogonalPlaneCross[0]
            y = orthogonalPlaneCross[1]
            z = a
        elif axis == 'x':
            axis2_name = 'y'
            axis3_name = 'z'
            y = orthogonalPlaneCross[0]
            z = orthogonalPlaneCross[1]
            x = a
        elif axis == 'y':
            axis2_name = 'x'
            axis3_name = 'z'
            x = orthogonalPlaneCross[0]
            z = orthogonalPlaneCross[1]
            y = a
        else:
            raise ValueError('Bad axis', axis)

        if objectRef:
            x -= self.x0
            y -= self.y0
            z -= self.z0

        if _type == 'potential':
            x, y, z, v = self.v_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
            unit = '[$J.kg^{-1}$]'
            _name = 'V'
        elif _type == 'acceleration':
            if ax is None or ay is None or az is None:
                x, y, z, ax, ay, az = self.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
            unit = '[$m.s^{-2}$]'
            _name = 'a'
            if acomp == 'x':
                v = ax
            elif acomp == 'y':
                v = ay
            elif acomp == 'z':
                v = az
            else:
                raise ValueError('Bad acomp!')
        elif _type == 'GGT':
            if Txx is None or Txy is None or Txz is None or Tyy is None or Tyz is None or Tzz is None:
                x, y, z, Txx, Txy, Txz, Tyy, Tyz, Tzz = self.T_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
            unit = '[$s^{-2}$]'
            _name = 'T'
            if Tcomp == 'xx':
                v = Txx
            elif Tcomp == 'xy':
                v = Txy
            elif Tcomp == 'xz':
                v = Txz
            elif Tcomp == 'yy':
                v = Tyy
            elif Tcomp == 'yz':
                v = Tyz
            elif Tcomp == 'zz':
                v = Tzz
            else:
                raise ValueError('Bad Tcomp!')
        else:
            raise ValueError('Bad type!')

        if pltAxis is None:
            pltAxis = plt.subplot(111)
        if yscale == 'log':
            v = np.abs(v)
        pltAxis.plot(a, v, color = 'black')
        if not hold:
            if log:
                pltAxis.set_xscale('log')
            pltAxis.set_yscale(yscale)
            pltAxis.set_xlabel(axis + ' [m]')
            laborth = axis2_name + "=" + str(orthogonalPlaneCross[0]) + ", " + axis3_name + "=" + str(orthogonalPlaneCross[1])
            pltAxis.set_ylabel(_name + r'$(' + laborth+ ', ' + axis + ')$ ' + unit)
            plt.show(block = True)

        if _type == 'potential':
            return a, v
        elif _type == 'acceleration':
            return a, ax, ay, az
        elif _type == 'GGT':
            return a, Txx, Txy, Txz, Tyy, Tyz, Tzz
        else:
            raise ValueError('Bad type')


    def plt_horizontal_aslice(self, xmin, xmax, ymin, ymax, z, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, ny = 200, nr = 100, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, object2 = None, fast = False):
        """
        Plot potential in a horizontal slice (in case of a cylinder, normal to cylinder main axis)
        """
        if fast and not self.kind in ['cylinder', 'ball']:
            print("gravity.solid.plt_horizontal_aslice: Warning. Fast computation for cylinder and ball only...")
            fast = False
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

            if not fast:
                ax, ay, az = self.plt_aSlice('(x,y)', xmin, xmax, ymin, ymax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = (nx, ny), orthogonalCross = z, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, ax = ax, ay = ay, az = az)
            else:
                ax, ay, az = self.plt_horizontal_aslice_cylindricalSymmetry(xmin, xmax, ymin, ymax, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, nx = nx, ny = ny, nr = nr, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, ax = ax, ay = ay, az = az)

            if self.kind in ['ball', 'cylinder']:
                plots.pltDiskProjection([self.x0, self.y0], self.radius, hollow = self.hollow, innerRadius = self.innerRadius, color = 'black', linestyle = '--', axis = locax)
                if object2 is not None:
                    plots.pltDiskProjection([object2.x0, object2.y0], object2.radius, hollow = object2.hollow, innerRadius = object2.innerRadius, color = 'black', linestyle = '-.', axis = locax)
            locax.set_xlim(xmin, xmax)
            locax.set_ylim(ymin, ymax)
            locax.set_ylabel('y [m]')
            if iplot == 3:
                locax.set_xlabel('x [m]')
            #else:
            #    locax.set_xticklabels([])
            iplot += 1
        #plt.suptitle(r'$a(' + laborth+ ', z)$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0, hspace = 0.05)
        plt.show(block = True)


    def plt_horizontal_aslice_cylindricalSymmetry(self, xmin, xmax, ymin, ymax, z, acomp = None, pltAxis = None, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, ny = 200, nr = 100, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, ax = None, ay = None, az = None, hold = False):
        if ax is None or ay is None or az is None:
            #first, compute radially
            r = np.linspace(self.x0, max(self.x0-xmin, self.x0+xmax, self.y0-ymin, self.y0+ymax), nr)
            x1 = r
            y1 = 0
            x1, y1, z1, ax1, ay1, az1 = self.a_grid(x1, y1, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)

            #then, assume cylindrical symmetry to populate the entire ring between both cylinders
            x = np.linspace(xmin, xmax, nx)
            y = np.linspace(ymin, ymax, ny)
            xp, yp = np.meshgrid(x, y)
            rp = np.sqrt((xp-self.x0)**2 + (yp-self.y0)**2)
            thetap = np.arctan2(yp, xp)
            fx = interp1d(r, ax1, kind = 'linear', bounds_error = False, fill_value = 0)
            fy = interp1d(r, ay1, kind = 'linear', bounds_error = False, fill_value = 0)
            fz = interp1d(r, az1, kind = 'linear', bounds_error = False, fill_value = 0)
            #ax = np.cos(thetap) * fx(rp) - np.sin(thetap) * fy(rp)
            #ay = np.sin(thetap) * fx(rp) + np.cos(thetap) * fy(rp)
            ax = np.cos(thetap) * fx(rp) #fx(rp) is ax taken along the x-axis, i.e. the (signed) norm of the acceleration; this is exactly like the full rotation above, but with the symmetry assumption y->-y explicit
            ay = np.sin(thetap) * fx(rp)
            az = fz(rp)
            #careful to lines vs rows...
            ax = np.transpose(ax)
            ay = np.transpose(ay)
            az = np.transpose(az)

        unit = '[$m.s^{-2}$]'
        if acomp == 'x':
            v = ax
            _name = '$a_x$'
        elif acomp == 'y':
            v = ay
            _name = '$a_y$'
        elif acomp == 'z':
            v = az
            _name = '$a_z$'
        else:
            raise ValueError('Bad acomp!')

        if pltAxis is None:
            pltAxis = plt.subplot(111)
        if log:
            v_slice = np.abs(v)
            _m = np.min(v_slice[np.isfinite(v_slice)])
            _M = np.max(v_slice[np.isfinite(v_slice)])
            if _m == 0:
                print("WARNING! solid.plt_vatSlice: min value is 0 (colorbar). Setting it to 0.01 of max")
                _m = 0.01 * _M
            nan = np.where((np.isfinite(v_slice) == False) | np.isnan(v_slice))
            v_slice[nan] = _m
            norm = LogNorm(vmin = _m, vmax = _M)
            #norm = LogNorm(vmin = np.min(v_slice), vmax = np.max(v_slice))
        else:
            v_slice = v
            norm = None
        v_slice = np.flipud(np.transpose(v_slice)) #v_grid output coord#1 as lines and coord#2 as columns, but we want the opposite (thence transpose), and then imshow shows first lines on top, but we want them at the bottom, thence flipud
        im = pltAxis.imshow(v_slice, cmap = 'jet', aspect = 'auto', extent = [xmin, xmax, ymin, ymax], norm = norm)
        #pltAxis.set_xlabel(axis1)
        #pltAxis.set_ylabel(axis2)
        #pltAxis.set_title("Potential " + axis3 + "=" + str(orthogonalCross))
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel(_name + unit)
        if not hold:
            pltAxis.set_xlabel(axis1)
            pltAxis.set_ylabel(axis2)
            pltAxis.set_title("Potential " + axis3 + "=" + str(orthogonalCross))
            plt.show(block = True)
            
        return ax, ay, az

    def plt_vertical_aslice(self, xmin, xmax, zmin, zmax, y, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, nz = 200, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, object2 = None):
        """
        Plot potential in a horizontal slice (normal to cylinder main axis)
        """
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

            ax, ay, az = self.plt_aSlice('(x,z)', xmin, xmax, zmin, zmax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = (nx, nz), orthogonalCross = y, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, ax = ax, ay = ay, az = az)
            if self.kind == 'ball':
                plots.pltDiskProjection([self.x0, self.y0], self.radius, hollow = self.hollow, innerRadius = self.innerRadius, color = 'black', linestyle = '--', axis = locax)
            elif self.kind == 'cylinder':
                plots.pltRectangleProjection([self.x0, self.z0], 2*self.radius, self.height, hollow = self.hollow, innerWidth = 2*self.innerRadius, color = 'black', linestyle = '--')
            else:
                raise NotImplementedError()
                
            if object2 is not None:
                if object2.kind == 'ball':
                    plots.pltDiskProjection([object2.x0, object2.y0], object2.radius, hollow = object2.hollow, innerRadius = object2.innerRadius, color = 'black', linestyle = '-.', axis = locax)
                elif object2.kind == 'cylinder':
                    plots.pltRectangleProjection([object2.x0, object2.z0], 2*object2.radius, object2.height, hollow = object2.hollow, innerWidth = 2*object2.innerRadius, color = 'black', linestyle = '-.')
                else:
                    raise NotImplementedError()

            locax.set_xlim(xmin, xmax)
            locax.set_ylim(zmin, zmax)
            locax.set_ylabel('z [m]')
            if iplot == 3:
                locax.set_xlabel('x [m]')
            #else:
            #    locax.set_xticklabels([])
            iplot += 1
        #plt.suptitle(r'$a(' + laborth+ ', y)$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0, hspace = 0.05)
        plt.show(block = True)

    def plt_vSlice(self, plane, _min1, _max1, _min2, _max2, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, pltAxis = None, objectRef = True, n = (100, 100), orthogonalCross = 0., log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, hold = False):
        return self.plt_vatSlice('potential', plane, _min1, _max1, _min2, _max2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, pltAxis = pltAxis, objectRef = objectRef, n = n, orthogonalCross = orthogonalCross, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = hold)

    def plt_aSlice(self, plane, _min1, _max1, _min2, _max2, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, acomp = None, pltAxis = None, objectRef = True, n = (100, 100), orthogonalCross = 0., log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, hold = False, ax = None, ay = None, az = None):
        return self.plt_vatSlice('acceleration', plane, _min1, _max1, _min2, _max2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = acomp, pltAxis = pltAxis, objectRef = objectRef, n = n, orthogonalCross = orthogonalCross, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = hold, ax = ax, ay = ay, az = az)

    def plt_TSlice(self, plane, _min1, _max1, _min2, _max2, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, Tcomp = None, pltAxis = None, objectRef = True, n = (100, 100), orthogonalCross = 0., log = False, v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False, hold = False, Txx = None, Txy = None, Txz = None, Tyy = None, Tyz = None, Tzz = None):
        return self.plt_vatSlice('GGT', plane, _min1, _max1, _min2, _max2, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, Tcomp = Tcomp, pltAxis = pltAxis, objectRef = objectRef, n = n, orthogonalCross = orthogonalCross, log = log, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose, hold = hold, Txx = Txx, Txy = Txy, Txz = Txz, Tyy = Tyy, Tyz = Tyz, Tzz = Tzz)
        
    def plt_vatSlice(self, _type, plane, _min1, _max1, _min2, _max2, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, acomp = None, Tcomp = None, pltAxis = None, objectRef = True, n = (100, 100), orthogonalCross = 0., log = False, method = 'tplquad', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False, hold = False, ax = None, ay = None, az = None, Txx = None, Txy = None, Txz = None, Tyy = None, Tyz = None, Tzz = None):
        """
        Plot slice of potential. Use plane parallel to Cartesian reference frame. Should be possible to have more general slicing, perhaps numpy or matplotlib provide tools for that, but I want to stay simple here...
        plane -- (x,y) or (x,z) or (y,z)
        orthogonalCross -- value at which plane crosses 3rd axis
        """
        if _min1 >= _max1 or _min2 >= _max2:
            raise ValueError("Need min < max")
        a1 = np.linspace(_min1, _max1, n[0])
        a2 = np.linspace(_min2, _max2, n[1])
        if plane == '(x,y)':
            axis1 = 'x'
            axis2 = 'y'
            axis3 = 'z'
            x = a1
            y = a2
            z = orthogonalCross
            xp, yp = np.meshgrid(x, y)
        elif plane == '(x,z)':
            axis1 = 'x'
            axis2 = 'z'
            axis3 = 'y'
            x = a1
            z = a2
            y = orthogonalCross
            xp, zp = np.meshgrid(x, z)
        elif plane == '(y,z)':
            axis1 = 'y'
            axis2 = 'z'
            axis3 = 'x'
            y = a1
            z = a2
            x = orthogonalCross
            yp, zp = np.meshgrid(y, z)
        else:
            raise ValueError("Bad plane")

        if objectRef:
            x -= self.x0
            y -= self.y0
            z -= self.z0

        if plane == '(x,y)':
            m1 = x[0]
            M1 = x[-1]
            m2 = y[0]
            M2 = y[-1]
            #p1, p2 = np.meshgrid(x, y)
        elif plane == '(x,z)':
            m1 = x[0]
            M1 = x[-1]
            m2 = z[0]
            M2 = z[-1]
            #p1, p2 = np.meshgrid(x, z)
        elif plane == '(y,z)':
            m1 = y[0]
            M1 = y[-1]
            m2 = z[0]
            M2 = z[-1]
            #p1, p2 = np.meshgrid(y, z)
        else:
            raise ValueError("Bad plane")

        if _type == 'potential':
            x, y, z, v = self.v_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
            unit = '[$J.kg^{-1}$]'
            _name = 'V'
        elif _type == 'acceleration':
            if ax is None or ay is None or az is None:
                x, y, z, ax, ay, az = self.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose)
            unit = '[$m.s^{-2}$]'
            if acomp == 'x':
                v = ax
                _name = '$a_x$'
            elif acomp == 'y':
                v = ay
                _name = '$a_y$'
            elif acomp == 'z':
                v = az
                _name = '$a_z$'
            else:
                raise ValueError('Bad acomp!')
        elif _type == 'GGT':
            if Txx is None or Txy is None or Txz is None or Tyy is None or Tyz is None or Tzz is None:
                x, y, z, Txx, Txy, Txz, Tyy, Tyz, Tzz = self.T_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
            unit = '[$s^{-2}$]'
            _name = 'T'
            if Tcomp == 'xx':
                v = Txx
                _name = '$T_{xx}$'
            elif Tcomp == 'xy':
                v = Txy
                _name = '$T_{xy}$'
            elif Tcomp == 'xz':
                v = Txz
                _name = '$T_{xz}$'
            elif Tcomp == 'yy':
                v = Tyy
                _name = '$T_{yy}$'
            elif Tcomp == 'yz':
                v = Tyz
                _name = '$T_{yz}$'
            elif Tcomp == 'zz':
                v = Tzz
                _name = '$T_{zz}$'
            else:
                raise ValueError('Bad Tcomp!')
        else:
            raise ValueError('Bad type!')

        if pltAxis is None:
            pltAxis = plt.subplot(111)
        if log:
            v_slice = np.abs(v)
            _m = np.min(v_slice[np.isfinite(v_slice)])
            _M = np.max(v_slice[np.isfinite(v_slice)])
            if _m == 0:
                print("WARNING! solid.plt_vatSlice: min value is 0 (colorbar). Setting it to 0.01 of max")
                _m = 0.01 * _M
            nan = np.where((np.isfinite(v_slice) == False) | np.isnan(v_slice))
            v_slice[nan] = _m
            norm = LogNorm(vmin = _m, vmax = _M)
            #norm = LogNorm(vmin = np.min(v_slice), vmax = np.max(v_slice))
        else:
            v_slice = v
            norm = None
        v_slice = np.flipud(np.transpose(v_slice)) #v_grid output coord#1 as lines and coord#2 as columns, but we want the opposite (thence transpose), and then imshow shows first lines on top, but we want them at the bottom, thence flipud
        im = pltAxis.imshow(v_slice, cmap = 'jet', aspect = 'auto', extent = [m1, M1, m2, M2], norm = norm)
        #pltAxis.set_xlabel(axis1)
        #pltAxis.set_ylabel(axis2)
        #pltAxis.set_title("Potential " + axis3 + "=" + str(orthogonalCross))
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel(_name + unit)
        if not hold:
            pltAxis.set_xlabel(axis1)
            pltAxis.set_ylabel(axis2)
            pltAxis.set_title("Potential " + axis3 + "=" + str(orthogonalCross))
            plt.show(block = True)

        if _type == 'potential':
            return v
        elif _type == 'acceleration':
            return ax, ay, az
        elif _type == 'GGT':
            return Txx, Txy, Txz, Tyy, Tyz, Tzz
        else:
            raise ValueError('Bad type')


    def plt_rhoSlice(self, plane, _min1, _max1, _min2, _max2, n = (100, 100), orthogonalCross = 0., hold = False):
        if _min1 >= _max1 or _min2 >= _max2:
            raise ValueError("Need min < max")
        a1 = np.linspace(_min1, _max1, n[0])
        a2 = np.linspace(_min2, _max2, n[1])
        if plane == '(x,y)':
            x, y, z, rho = self.mapRho(_min1, _max1, n[0], _min2, _max2, n[1], orthogonalCross, orthogonalCross, 1)
            axis1 = 'x'
            axis2 = 'y'
            axis3 = 'z'
            m1 = x[0]
            M1 = x[-1]
            m2 = y[0]
            M2 = y[-1]
            xp, yp = np.meshgrid(x, y)
        elif plane == '(x,z)':
            x, y, z, rho = self.mapRho(_min1, _max1, n[0], orthogonalCross, orthogonalCross, 1, _min2, _max2, n[1], )
            axis1 = 'x'
            axis2 = 'z'
            axis3 = 'y'
            m1 = x[0]
            M1 = x[-1]
            m2 = z[0]
            M2 = z[-1]
            xp, zp = np.meshgrid(x, z)
        elif plane == '(y,z)':
            x, y, z, rho = self.mapRho(orthogonalCross, orthogonalCross, 1, _min1, _max1, n[0], _min2, _max2, n[1])
            axis1 = 'y'
            axis2 = 'z'
            axis3 = 'x'
            m1 = y[0]
            M1 = y[-1]
            m2 = z[0]
            M2 = z[-1]
            yp, zp = np.meshgrid(y, z)
        else:
            raise ValueError("Bad plane")

        rho = np.flipud(np.transpose(rho))
        im = plt.imshow(rho, cmap = 'jet', aspect = 'auto', extent = [m1, M1, m2, M2])
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("Density [kg.m$^{-3}$]")
        plt.xlabel(axis1)
        plt.ylabel(axis2)
        plt.title("Density " + axis3 + "=" + str(orthogonalCross))
        if not hold:
            plt.show(block = True)



    
def scalar2array(s):
    if np.size(s) > 1:
        return s
    else:
        return np.array([s])

    
