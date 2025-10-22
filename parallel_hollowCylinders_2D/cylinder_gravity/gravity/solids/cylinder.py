import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.integrate import simpson
from scipy.interpolate import interp1d
#try:
#    from skmonaco import mcquad #scikit-monaco
#except:
#    print("WARNING! scikit-monaco not found, and seems obsolete. Cannot perform Monte Carlo integration (or need a new package)")
from scipy.constants import G
from scipy.integrate import tplquad
from scipy.special import lpn, factorial
import time
from gravity.solids import solid, plots
from gravity.solids.progressETA import progress

class cylinder(solid.solid):
    def __init__(self, radius, height, center = [0.,0.,0.], origin = [0.,0.,0.], density = 1, originType = 'centered', innerRadius  = 0):
        """
        innerRadius -- if >0, defines a hollow cylinder
        """
        super(cylinder, self).__init__(density, center = center, origin = origin)
        self.kind = 'cylinder'
        self.radius = radius
        self.innerRadius = innerRadius
        self.hollow = (self.innerRadius > 0)
        self.height = height
        self.originType = originType
        self.setOrigin(origin) #done in solid class, but should be updated here
        #if originType == 'centered':
        #    self.hmin = self.z0 - 0.5 * height
        #    self.hmax = self.z0 + 0.5 * height
        #elif originType == 'low':
        #    self.hmin = self.z0
        #    self.hmax = height
        #else:
        #    raise NotImplementedError()
        self.tplquad_xmin = self.x0 - self.radius #lower boundary for x for tplquad integration
        self.tplquad_xmax = self.x0 + self.radius #upper boundary for x for tplquad integration

    def setOrigin(self, origin):
        self.x0 = self.xc + origin[0]
        self.y0 = self.yc + origin[1]
        self.z0 = self.zc + origin[2]
        if self.kind == 'undefined':
            #can happen at first call from solid
            return
        elif self.kind == 'cylinder':
            if self.originType == 'centered':
                self.hmin = self.z0 - 0.5 * self.height
                self.hmax = self.z0 + 0.5 * self.height
            elif self.originType == 'low':
                self.hmin = self.z0
                self.hmax = self.height
            #elif self.originType is None:
            #    pass
            else:
                raise NotImplementedError()
            self.tplquad_xmin = self.x0 - self.radius #lower boundary for x for tplquad integration
            self.tplquad_xmax = self.x0 + self.radius #upper boundary for x for tplquad integration
        else:
            raise ValueError(self.kind + ' solid kind not allowed in cylinder object')
        
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
        return self.hmin

    def tplquad_zmax_fun(self, x, y):
        """upper boundary for z for tplquad integration (rfun in terms of tplquad doc)"""
        return self.hmax
        
    def cmpVolume(self):
        if not self.hollow:
            return np.pi * self.radius**2 * self.height
        else:
            return np.pi * (self.radius**2 - self.innerRadius**2) * self.height

    def inner(self, xyzmin, xyzmax, strictBorder = True):
        raise NotImplementedError("Now called inside()")
        
    def inside(self, xyzmin, xyzmax, strictBorder = True, gridType = None):
        """
        Check if a small volume (or an ensemble of small volumes) is inside or outside the cylinder. Return 1 for volume(s) inside the cylinder, 0 for others
        xyzmin -- (xleft, yleft, zmin)
        xyzmax -- (xright, yright, zmax)
        strictBorder -- if set, a small cube is said to be inside the cylinder if it is entirely inside. Otherwise, said to be inside even if it is halfway out.
        """
        #gridType = 'Cartesian'
        if gridType is None:
            raise TypeError("Please set gridType (this error message should be temporary, until all calls are cleaned up in the code... and a default value is provided)")
        if gridType == 'polar':
            rleft = xyzmin[0]
            thleft = xyzmin[1]
            rright = xyzmax[0]
            thright = xyzmax[1]
            xleft = rleft * np.cos(thleft)
            yleft = rleft * np.sin(thleft)
            xright = rright * np.cos(thright)
            yright = rright * np.sin(thright)
            zmin = xyzmin[2] #- self.z0
            zmax = xyzmax[2] #- self.z0
        else:
            xleft = xyzmin[0] - self.x0
            yleft = xyzmin[1] - self.y0
            zmin = xyzmin[2] #- self.z0
            xright = xyzmax[0] - self.x0
            yright = xyzmax[1] - self.y0
            zmax = xyzmax[2] #- self.z0
        
        rleft = np.sqrt(xleft**2 + yleft**2)
        rright = np.sqrt(xright**2 + yright**2)
        rmin = np.minimum(rleft, rright)
        rmax = np.maximum(rleft, rright)

        if np.size(xleft) == 1:
            if strictBorder:
                if rmax <= self.radius and rmin >= self.innerRadius and zmax <= self.hmax and zmin >= self.hmin:
                    return 1
                else:
                    return 0
            else:
                if (rmin <= self.radius and rmax >= self.innerRadius) and ((zmin <= self.hmax and zmin >= self.hmin) or (zmax >= self.hmin and zmax <= self.hmax)):
                    return 1
                else:
                    return 0
        else:
            output = np.zeros(np.shape(xleft))
            if strictBorder:
                inner = np.where((rmax <= self.radius) & (rmin >= self.innerRadius) & (zmax <= self.hmax) & (zmin >= self.hmin))#[0]
            else:
                inner = np.where(((rmin <= self.radius) & (rmax >= self.innerRadius)) & (((zmin <= self.hmax) & (zmin >= self.hmin)) | ((zmax >= self.hmin) & (zmax <= self.hmax))))#[0]
            output[inner] = 1
            return output

    def innerPoint(self, x_in, y_in, z_in):
        raise NotImplementedError("Now called insidePoint()")
        
    def insidePoint(self, x_in, y_in, z_in, gridType = None):
        """
        Check if a point (or an ensemble of points) is inside or outside the cylinder. Return 1 for point(s) inside the cylinder, 0 for others
        """
        #gridType = 'Cartesian'
        if gridType is None:
            raise TypeError("Please set gridType (this error message should be temporary, until all calls are cleaned up in the code... and a default value is provided)")
        if gridType == 'polar':
            r = x_in
            th = y_in
            x = r * np.cos(th)
            y = r * np.sin(th)
            z = z_in #- self.z0
        else:
            x = x_in - self.x0
            y = y_in - self.y0
            z = z_in #- self.z0
        if np.size(x) == 1:
            if np.sqrt(x**2 + y**2) <= self.radius and np.sqrt(x**2 + y**2) >= self.innerRadius and z <= self.hmax and z >= self.hmin:
                return 1
            else:
                return 0
        else:
            r = np.sqrt(x**2 + y**2)
            output = np.zeros(np.shape(x))
            inner = np.where((r <= self.radius) & (r >= self.innerRadius) & (z <= self.hmax) & (z >= self.hmax))#[0]
            output[inner] = 1
            return output

    def mkGrid(self, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, xmin = None, xmax = None, ymin = None, ymax = None, dr_rel = 0.01, dtheta_rel = 0.01, rmin = None, rmax = None, thmin = None, thmax = None, zmin = None, zmax = None, borders = True, _type = 'Cartesian'):
        """
        Make regular grid inside cylinder
        dx, dy, dz -- relative size (wrt to cylinder size) of volume elements
        """
        if _type == 'Cartesian':
            return self.mkCartesianGrid(dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, zmin = zmin, zmax = zmax, borders = borders)
        elif _type == 'polar':
            return self.mkPolarGrid(dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, rmin = rmin, rmax = rmax, thmin = thmin, thmax = thmax, zmin = zmin, zmax = zmax, borders = borders)
        else:
            raise ValueError("Bad type " + _type + ". Must Cartesian or polar")
        
    def mkCartesianGrid(self, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, xmin = None, xmax = None, ymin = None, ymax = None, zmin = None, zmax = None, borders = True):
        """
        Make regular grid inside cylinder
        dx, dy, dz -- relative size (wrt to cylinder size) of volume elements
        """
        #dx = self.radius * dx_rel
        #dy = self.radius * dy_rel
        #dz = self.height * dz_rel

        if xmin is None or xmax is None:
            dx = 2 * self.radius * dx_rel
            xspan = 2 * 1.01 * self.radius
            nx = int(xspan / dx) + 1
            x = np.linspace(-1.01 * self.radius, 1.01 * self.radius, nx) + self.x0
        else:
            dx = (xmax - xmin) * dx_rel
            nx = int((xmax - xmin) / dx + 1)
            x = np.linspace(xmin, xmax, nx)
        if ymin is None or ymax is None:
            dy = 2 * self.radius * dy_rel
            yspan = 2 * 1.01 * self.radius
            ny = int(yspan / dy) + 1
            y = np.linspace(-1.01 * self.radius, 1.01 * self.radius, ny) + self.y0
        else:
            if ymin != ymax:
                dy = (ymax - ymin) * dy_rel
            else:
                dy = 2 * self.radius * dy_rel
            ny = int((ymax - ymin) / dy + 1)
            y = np.linspace(ymin, ymax, ny)
        if zmin is None or zmax is None:
            dz = self.height * dz_rel
            zspan = self.hmax + 0.05 * self.height - (self.hmin - 0.05 * self.height)
            nz = int(zspan / dz) + 1
            z = np.linspace(self.hmin - 0.05 * self.height, self.hmax + 0.05 * self.height, nz)
        else:
            if zmin != zmax:
                dz = (zmax - zmin) * dz_rel
            else:
                dz = self.height * dz_rel
            #dz = (zmax - zmin) * dz_rel
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

    def mkPolarGrid(self, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, rmin = None, rmax = None, thmin = None, thmax = None, zmin = None, zmax = None, borders = True):
        """
        Make regular grid inside cylinder
        dx, dy, dz -- relative size (wrt to cylinder size) of volume elements
        """
        dr = (self.radius - self.innerRadius) * dr_rel
        dth = 2. * np.pi * dtheta_rel
        dz = self.height * dz_rel

        #define lower (left) limits
        if rmin is None or rmax is None:
            rspan = self.radius - self.innerRadius
            nr = int(rspan / dr) + 1
            r = np.linspace(self.innerRadius, self.radius, nr) #+ self.x0
        else:
            nr = int((rmax - rmin) / dr + 1)
            r = np.linspace(rmin, rmax, nr)
        if thmin is None or thmax is None:
            thspan = 2. * np.pi
            nth = int(thspan / dth) + 1
            theta = np.linspace(0, 2. * np.pi, nth) #+ self.y0
        else:
            nth = int((thmax - thmin) / dth + 1)
            theta = np.linspace(thmin, thmax, nth)
        if zmin is None or zmax is None:
            zspan = self.hmax + 0.05 * self.height - (self.hmin - 0.05 * self.height)
            nz = int(zspan / dz) + 1
            z = np.linspace(self.hmin - 0.05 * self.height, self.hmax + 0.05 * self.height, nz)
        else:
            nz = int((zmax - zmin) / dz + 1)
            z = np.linspace(zmin, zmax, nz)

        #make sure that last bin stops at max. So just need to say that we've just defined "left" borders, with one extra that we remove now
        rl = r[:-1]
        thl = theta[:-1]
        zl = z[:-1]
        nr = np.size(rl)
        nth = np.size(thl)
        nz = np.size(zl)
        
        if borders:
            r = rl + 0.5 * dr
            rr = rl + dr
            theta = thl + 0.5 * dth
            thr = thl + dth
            z = zl + 0.5 * dz
            zr = zl + dz
            return r, rl, rr, nr, theta, thl, thr, nth, z, zl, zr, nz
        else:
            r = rl + 0.5 * dr
            theta = thl + 0.5 * dth
            z = zl + 0.5 * dz
            return r, nr, theta, nth, z, nz

        
    def inertiaMoment(self, i, j, k, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, method = 'simps', mc_npoints = 1e6, memoryFriendly = False, verbose = True, fullNumeric = False):
        """
        Compute ijk inertia moment (Int[rho*x^i*y^j*z^k dxdydz])
        method -- tplquad or simps
        """
        fullNumeric = True
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
            f1 = (self.radius**(i+j+2) - self.innerRadius**(i+j+2)) / (i+j+2)
            f2 = (self.hmax**(k+1) - self.hmin**(k+1)) / (k+1)
            I = self.density * f1 * f2 * Iij

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
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        
        if timeit:
            st = time.time()
        dU = lambda xyz: self.infinitesimalVElementPotential(xyz[0], xyz[1], xyz[2], X, Y, Z)
        U, err = mcquad(dU, npoints, [self.radius, self.radius, 0.5 * self.height], [-self.radius, -self.radius, -0.5 * self.height])
        if np.abs(err) >= 1e-3 * np.abs(U):
            print("WARNING! mcquad error is more than 1e-3 integral estimate...", U, err)
        U *= G
        if timeit:
            et = time.time()
            return (U, et - st)
        return U

    # def cmpPotential_XYZ_inertiaMoments(self, XYZ, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, im_method = 'simps', memoryFriendly = False, timeit = False):
    #     """
    #     see MIC-NT-SY-0-6002-OCA
    #     XYZ -- (X, Y, Z)
    #     memoryFriendly -- use for loops instead of arrays. Much slower, but can be more memory efficient if dx, dy and dz are very small
    #     """
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
    #     #V0 = mass / R
    #     V0 = 1. / R
    #     V3 = (X * self.I100 + Y * self.I010 + Z * self.I001) / R**3
    #     V5 = (3 * (X * Y * self.I110 + X * Z * self.I101 + Y * Z * self.I011) + 0.5 * ((3. * X**2 - R**2) * self.I200 + (3. * Y**2 - R**2) * self.I020 + (3. * Z**2 - R**2) * self.I002)) / R**5
    #     V7 = ((15. * X**2 - 3. * R**2) * (Y * self.I210 + Z * self.I201) + (15. * Y**2 - 3. * R**2) * (X * self.I120 + Z * self.I021) + (15. * Z**2 - 3. * R**2) * (X * self.I102 + Z * self.I012) + 30. * X * Y * Z * self.I111 + (5. * X**2 - 3. * R**2) * X * self.I300 + (5. * Y**2 - 3. * R**2) * Y * self.I030 + (5. * Z**2 - 3. * R**2) * X * self.I003 ) / (2. * R**7)
    #     V = G * mass * (V0 + V3 + V5 + V7)
    #     #V = G * V0 #Gm/R
    #     if timeit:
    #         et = time.time()
    #         return (V, et - st)
    #     return V


    def cmpOnAxisPotential_ana(self, z):
        """
        Analytic potential along the main axis of the cylinder (z), outside the cylinder, above it, when the origin of z is taken at the base of the cylinder
        """
        if self.hmin != self.z0:
            raise ValueError("To compute analytic potential, need cylinder to pbe based on z0 (not centered)")
        zz = z - self.z0
        bd = np.where(zz < self.height)
        if np.size(bd) > 0:
            raise ValueError("Can compute analytical potential only above the cylinder.")
        t1 = 2. * np.pi * G * self.density * zz * self.height
        t2 = -np.pi * G *self.density * self.height ** 2
        t3 = (self.height - zz) * np.sqrt(self.radius**2 + (zz - self.height)**2)
        t4 = zz * np.sqrt(self.radius**2 + zz**2)
        t5 = -self.radius**2 * np.log((zz - self.height + np.sqrt(self.radius**2 + (zz - self.height)**2)) / (z + np.sqrt(self.radius**2 + zz**2)))
        return -(t1 + t2 - np.pi * G *self.density * (t3 + t4 + t5))


    def cmpPotential_ana_multipolar(self, XYZ, pmax = 15, raiseWarning = True):
        """
        Lockerbie et al 1993
        Only Newtonian!
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        if np.size(X) > 1:
            n = np.size(X)
            v = np.zeros(n)
            for i in range(n):
                v[i] = self.cmpPotential_ana_multipolar_XYZ([X[i], Y[i], Z[i]], pmax = pmax, raiseWarning = raiseWarning)
            return v
        else:
            return self.cmpPotential_ana_multipolar_XYZ(XYZ, pmax = pmax, raiseWarning = raiseWarning)
            
    def cmpPotential_ana_multipolar_XYZ(self, XYZ, pmax = 15, raiseWarning = True):
        """
        Lockerbie et al 1993
        Only Newtonian!
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        if np.size(X) > 1 or np.size(Y) > 1 or np.size(Z) > 1:
            raise NotImplementedError('Ooops, X etc must be scalars')
        if not type(X) in [int, float, np.float64]:
            X = X[0]
        if not type(Y) in [int, float, np.float64]:
            Y = Y[0]
        if not type(Z) in [int, float, np.float64]:
            Z = Z[0]
        r = np.sqrt(X**2 + Y**2)
        R = np.sqrt(r**2 + Z**2)
        cosTheta = Z / R
        M = self.cmpMass()
        pnx, dpnx = lpn(2 * pmax, cosTheta) #LegendreP[2p, cos(theta)]
        vp = 0
        for p in range(pmax+1):
            k2 = k2p(p, self.innerRadius, self.radius, 0.5 * self.height) #0.5*height because here I define the cylinder from -height/2 to height/2 but Lockerbie from -L to L
            P2p = pnx[2*p]
            #vp += (k2 * P2p  * factorial(2*p) * (0.5 * self.height / R)**(2 * p))
            vp += (k2 * P2p  * (0.5 * self.height / R)**(2 * p))
            if self.height / R >= 1 and raiseWarning:
                print("WARNING! cylinder.cmpPotential_ana_multipolar_XYZ: l/R >=1, series will not converge! " + str(self.height / R))
            
        vp *= (-G * M) / R
        return vp

    def cmpAcceleration_ana_multipolar(self, XYZ, pmax = 15, raiseWarning = True):
        """
        Lockerbie et al 1993
        Only Newtonian!
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        if np.size(X) > 1:
            n = np.size(X)
            ax = np.zeros(n)
            ay = np.zeros(n)
            az = np.zeros(n)
            for i in range(n):
                ax[i], ay[i], az[i] = self.cmpAcceleration_ana_multipolar_XYZ([X[i], Y[i], Z[i]], pmax = pmax, raiseWarning = raiseWarning)
            return ax, ay, az
        else:
            return self.cmpAcceleration_ana_multipolar_XYZ(XYZ, pmax = pmax, raiseWarning = raiseWarning)
    
    def cmpAcceleration_ana_multipolar_XYZ(self, XYZ, pmax = 15, raiseWarning = True):
        """
        Lockerbie et al 1993
        Only Newtonian!
        """
        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        if np.size(X) > 1 or np.size(Y) > 1 or np.size(Z) > 1:
            raise NotImplementedError('Ooops, X etc must be scalars')
        if not type(X) in [int, float, np.float64]:
            X = X[0]
        if not type(Y) in [int, float, np.float64]:
            Y = Y[0]
        if not type(Z) in [int, float, np.float64]:
            Z = Z[0]
        phi = np.arctan2(Y, X)
        r = np.sqrt(X**2 + Y**2)
        R = np.sqrt(r**2 + Z**2)
        cosTheta = Z / R
        sinTheta = r / R
        M = self.cmpMass()
        pnx, dpnx = lpn(2 * pmax + 1, cosTheta) #LegendreP[2p, cos(theta)]
        ar = 0
        az = 0
        for p in range(pmax+1):
            k2 = k2p(p, self.innerRadius, self.radius, 0.5 * self.height) #0.5*height because here I define the cylinder from -height/2 to height/2 but Lockerbie from -L to L
            P2p = pnx[2*p]
            P2p1 = pnx[2*p+1]
            if np.abs(sinTheta) > 1e-6:
                arp = (2 * p + 1) * (k2 * (0.5 * self.height / R)**(2 * p)) / sinTheta * (cosTheta * P2p1 - P2p)
            else:
                arp = 0
            if self.height / R >= 1 and raiseWarning:
                print("WARNING! cylinder.cmpPotential_ana_multipolar_XYZ: l/R >=1, series will not converge! " + str(self.height / R))
            #print("--->", p, k2, self.height, R, self.height / R, (0.5 * self.height / R)**(2 * p), sinTheta, P2p1, P2p, arp, ar)
            ar += arp
            az += (2 * p + 1) * (k2 * P2p1  * (0.5 * self.height / R)**(2 * p)) #Eq. 30 of Lockerbie 1993
        ar *= G * M / R**2
        az *= G * M / R**2
        ax = ar * np.cos(phi)
        ay = ar * np.sin(phi)
        az *= -np.sign(Z - self.z0)
        #print(self.x0, self.y0, self.z0)
        return ax, ay, az

    def plt_vr(self, rmin, rmax, nr = 100, theta = 0, z = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, compareWAna = False, verbose = False, gridType = 'Cartesian'):
        """
        Plot potential as a function of radial distance from center of cylinder
        rmin, rmax - r limits [m]
        theta -- polar angle coordinate
        z -- altitude
        """
        if method == 'ana multipolar' and compareWAna:
            #no need to compare analytic with itself...
            compareWAna = False
        if rmax <= rmin:
            raise ValueError("rmin must be < rmax")
        if not log:
            r = np.linspace(rmin, rmax, nr)
        else:
            if rmin <= 0:
                raise ValueError("For log, rmin must be strictly >0")
            r = np.logspace(np.log10(rmin), np.log10(rmax), nr)
        if np.abs(theta) > 1e-6:
            x = r * np.cos(theta) - self.x0
            y = r * np.sin(theta) - self.y0
        else:
            x = r - self.x0
            y = 0

        x, y, z, v = self.v_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, gridType = gridType)
        if compareWAna:
            x, y, z, vAna = self.v_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = 'ana multipolar', im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose)
        if yscale == 'log':
            v = np.abs(v)
        plt.plot(r, v, color = 'black', label = method)
        if compareWAna:
            if yscale == 'log':
                vAna = np.abs(vAna)
            plt.plot(r, vAna, color = 'blue', linestyle = '--', label = 'ana (pmax=' + str(pmax) + ')')
            plt.legend()
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
        plt.tight_layout()
        plt.show(block = True)

        
    def plt_vz(self, zmin, zmax, nz = 100, xcross = 0, ycross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential as a function of height (parallel to axis)
        zmin, zmax - z limits [m]
        xcross, ycross -- x and y where potential is plotted
        """
        z, v = self.plt_vAlongAxis('z', zmin, zmax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, n = nz, orthogonalPlaneCross = (xcross, ycross), log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True)
        ylim = plt.ylim()
        if not log and self.hmin <= 0:
            plt.plot([self.hmin, self.hmin], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        plt.plot([self.hmax, self.hmax], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        suph = np.where(z - self.z0 > self.height)[0]
        if np.size(suph) > 0 and xcross == 0 and ycross == 0 and self.hmin == self.z0:
            vana = self.cmpOnAxisPotential_ana(z[suph])
            plt.plot(z[suph], vana, linestyle = '--')
        if log:
            plt.xscale('log')
        plt.yscale(yscale)
        plt.xlabel('z [m]')
        laborth = "x=" + str(xcross) + ", y=" + str(ycross)
        plt.ylabel(r'$V(' + laborth+ ', z)$ [$J.kg^{-1}$]')
        plt.tight_layout()
        plt.show(block = True)
        

    def plt_ar(self, rmin, rmax, nr = 100, theta = 0, z = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, compareWAna = False, verbose = False, gridType = 'Cartesian'):
        """
        Plot acceleration as a function of radial distance from center of cylinder
        rmin, rmax - r limits [m]
        theta -- polar angle coordinate
        z -- altitude
        """
        if method == 'ana multipolar' and compareWAna:
            #no need to compare analytic with itself...
            compareWAna = False
        if rmax <= rmin:
            raise ValueError("rmin must be < rmax")
        if not log:
            r = np.linspace(rmin, rmax, nr)
        else:
            if rmin <= 0:
                raise ValueError("For log, rmin must be strictly >0")
            r = np.logspace(np.log10(rmin), np.log10(rmax), nr)
        if np.abs(theta) > 1e-6:
            x = r * np.cos(theta) - self.x0
            y = r * np.sin(theta) - self.y0
        else:
            x = r - self.x0
            y = 0

        x, y, z, ax, ay, az = self.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, gridType = gridType)
        if compareWAna:
            x, y, z, axAna, ayAna, azAna = self.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = 'ana multipolar', im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose)
        if yscale == 'log':
            ax = np.abs(ax)
            ay = np.abs(ay)
            az = np.abs(az)
        if compareWAna:
            if yscale == 'log':
                axAna = np.abs(axAna)
                ayAna = np.abs(ayAna)
                azAna = np.abs(azAna)
            #plt.plot(r, vAna, color = 'blue', linestyle = '--', label = 'ana (pmax=' + str(pmax) + ')')
            #plt.legend()
        iplot = 1
        avec = [ax, ay, az]
        astr = ['$a_x$', '$a_y$', '$a_z$']
        if compareWAna:
            avecAna = [axAna, ayAna, azAna]
        while iplot < 4:
            if iplot == 1:
                axis = plt.subplot(311)
                locax = axis
            else:
                locax = plt.subplot(3,1,iplot, sharex = axis)
            v = avec[iplot-1]
            if compareWAna:
                vAna = avecAna[iplot-1]
            locax.plot(r, v, color = 'black')
            if compareWAna:
                locax.plot(r, vAna, color = 'blue', linestyle = '--', label = 'ana (pmax=' + str(pmax) + ')')
                if iplot == 1:
                    locax.legend()
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


    def plt_az(self, zmin, zmax, nz = 100, xcross = 0, ycross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot acceleration as a function of height (parallel to axis)
        zmin, zmax - z limits [m]
        xcross, ycross -- x and y where potential is plotted
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
            z, ax, ay, az = self.plt_aAlongAxis('z', zmin, zmax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = nz, orthogonalPlaneCross = (xcross, ycross), log = log, yscale = yscale, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, ax = ax, ay = ay, az = az)
            ylim = plt.ylim()
            if not log and self.hmin <= 0:
                locax.plot([self.hmin, self.hmin], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            locax.plot([self.hmax, self.hmax], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            suph = np.where(z - self.z0 > self.height)[0]
            #if iplot == 3 and np.size(suph) > 0 and xcross == 0 and ycross == 0 and self.hmin == self.z0:
            #    aana = self.cmpOnAxisAcceleration_ana(z[suph])
            #    locax.plot(z[suph], aana, linestyle = '--')
            if log:
                locax.set_xscale('log')
            locax.set_yscale(yscale)
            laborth = "x=" + str(xcross) + ", y=" + str(ycross)
            locax.set_ylabel(astr[iplot-1], fontsize = 13)
            if iplot == 3:
                locax.set_xlabel('z [m]')
            else:
                locax.set_xticklabels([])
            iplot += 1
        plt.suptitle(r'$a(' + laborth+ ', z)$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0, hspace = 0.05)
        plt.show(block = True)


    def plt_Tr(self, rmin, rmax, nr = 100, theta = 0, z = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, log = False, yscale = 'linear', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False, fname = None):
        """
        Plot GGT as a function of radial distance from center of cylinder
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
        if np.abs(theta) > 1e-6:
            x = r * np.cos(theta) - self.x0
            y = r * np.sin(theta) - self.y0
        else:
            x = r - self.x0
            y = 0

        x, y, z, Txx, Txy, Txz, Tyy, Tyz, Tzz = self.T_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose)
        trace = Txx + Tyy + Tzz
        if yscale == 'log':
            Txx = np.abs(Txx)
            Txy = np.abs(Txy)
            Txz = np.abs(Txz)
            Tyy = np.abs(Tyy)
            Tyz = np.abs(Tyz)
            Tzz = np.abs(Tzz)
            trace = np.abs(trace)

        iplot = 1
        avec = [Txx, Txy, Txz, None, Tyy, Tyz, trace, None, Tzz]
        astr = ['$T_{xx}$', '$T_{xy}$', '$T_{xz}$', '$T_{yx}$', '$T_{yy}$', '$T_{yz}$', 'Tr', '$T_{zy}$', '$T_{zz}$']
        while iplot < 10:
            #if iplot in [4,7,8]:
            if iplot in [4,8]:
                iplot += 1
                continue
            if iplot == 1:
                axis = plt.subplot(331)
                locax = axis
            else:
                locax = plt.subplot(3,3,iplot, sharex = axis)
            v = avec[iplot-1]
            locax.plot(r, v, color = 'black')
            ylim = plt.ylim()
            if not log and r[0] < 0:
                locax.plot([-self.radius, -self.radius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            locax.plot([self.radius, self.radius], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            if log:
                locax.set_xscale('log')
            locax.set_yscale(yscale)
            if iplot in [1,5,7,9]:
                locax.set_xlabel('r (m)', fontsize = 13)
            else:
                locax.set_xticklabels([])
            #locax.set_ylabel(astr[iplot-1] + r'$(r, z=' + str(z) + ')$ [$m.s^{-2}$]', fontsize = 13)
            locax.set_ylabel(astr[iplot-1], fontsize = 13)
            locax.tick_params(axis = 'both', which = 'major', labelsize = 13)
            iplot += 1
        plt.suptitle(r'$T(r, z=' + str(z) + ')$ [$s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        if fname is None:
            plt.show(block = True)
        else:
            plt.savefig(fname)
            plt.cla()
            plt.clf()

    def plt_Tz(self, zmin, zmax, nz = 100, xcross = 0, ycross = 0, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, log = False, yscale = 'linear', v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False):
        """
        Plot potential as a function of height (parallel to axis)
        zmin, zmax - z limits [m]
        xcross, ycross -- x and y where potential is plotted
        """
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
            z, Txx, Txy, Txz, Tyy, Tyz, Tzz = self.plt_TAlongAxis('z', zmin, zmax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, Tcomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = nz, orthogonalPlaneCross = (xcross, ycross), log = log, yscale = yscale, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, Txx = Txx, Txy = Txy, Txz = Txz, Tyy = Tyy, Tyz = Tyz, Tzz = Tzz)
            trace = Txx + Tyy + Tzz
            ylim = plt.ylim()
            if not log and self.hmin <= 0:
                locax.plot([self.hmin, self.hmin], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            locax.plot([self.hmax, self.hmax], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
            #suph = np.where(z - self.z0 > self.height)[0]
            #if iplot == 3 and np.size(suph) > 0 and xcross == 0 and ycross == 0 and self.hmin == self.z0:
            #    aana = self.cmpOnAxisAcceleration_ana(z[suph])
            #    locax.plot(z[suph], aana, linestyle = '--')
            if log:
                locax.set_xscale('log')
            locax.set_yscale(yscale)
            laborth = "x=" + str(xcross) + ", y=" + str(ycross)
            locax.set_ylabel(astr[iplot-1], fontsize = 13)
            if iplot in [1,5,9]:
                locax.set_xlabel('z [m]')
            else:
                locax.set_xticklabels([])
            iplot += 1
        #finally, plot trace
        locax = plt.subplot(3,3,7, sharex = axis)
        locax.plot(z, trace)
        ylim = plt.ylim()
        if not log and self.hmin <= 0:
            locax.plot([self.hmin, self.hmin], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        locax.plot([self.hmax, self.hmax], [ylim[0], ylim[1]], color = 'black', linestyle = ':')
        #suph = np.where(z - self.z0 > self.height)[0]
        #if iplot == 3 and np.size(suph) > 0 and xcross == 0 and ycross == 0 and self.hmin == self.z0:
        #    aana = self.cmpOnAxisAcceleration_ana(z[suph])
        #    locax.plot(z[suph], aana, linestyle = '--')
        if log:
            locax.set_xscale('log')
        locax.set_yscale(yscale)
        locax.set_ylabel("Tr", fontsize = 13)
        #... and conclude plot
        plt.suptitle(r'$T(' + laborth+ ', z)$ [$s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        plt.show(block = True)


        
    def plt_horizontal_vslice(self, xmin, xmax, ymin, ymax, z, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, ny = 200, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential in a horizontal slice (normal to cylinder main axis)
        """
        self.plt_vSlice('(x,y)', xmin, xmax, ymin, ymax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, n = (nx, ny), orthogonalCross = z, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True)
        theta = np.linspace(0, 2*np.pi, 500)
        xc = self.radius * np.cos(theta)
        yc = self.radius * np.sin(theta)
        if z - self.z0 <= self.hmax and z - self.z0 >= self.hmin:
            linestyle = '--'
        else:
            linestyle = ':'
        plt.plot(xc, yc, color = 'black', linestyle = linestyle)
        if self.hollow:
            xc = self.innerRadius * np.cos(theta)
            yc = self.innerRadius * np.sin(theta)
            if z - self.z0 <= self.hmax and z - self.z0 >= self.hmin:
                linestyle = '--'
            else:
                linestyle = ':'
            plt.plot(xc, yc, color = 'black', linestyle = linestyle)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show(block = True)

        
    def plt_vertical_vslice(self, xmin, xmax, zmin, zmax, y, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, nz = 200, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential in a horizontal slice (normal to cylinder main axis)
        """
        self.plt_vSlice('(x,z)', xmin, xmax, zmin, zmax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, objectRef = objectRef, n = (nx, nz), orthogonalCross = y, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, hold = True)
        #if y - self.y0 <= self.radius and y - self.y0 >= -self.radius:
        #    left = -np.sqrt(self.radius**2 - y**2)
        #    right = np.sqrt(self.radius**2 - y**2)
        #    linestyle = '--'
        #else:
        #    left = -self.radius
        #    right = self.radius
        #    linestyle = ':'
        linestyle = '--'
        plt.plot([left, right], [self.hmax, self.hmax], color = 'black', linestyle = linestyle)
        plt.plot([left, right], [self.hmin, self.hmin], color = 'black', linestyle = linestyle)
        plt.plot([left, left], [self.hmin, self.hmax], color = 'black', linestyle = linestyle)
        plt.plot([right, right], [self.hmin, self.hmax], color = 'black', linestyle = linestyle)
        if self.hollow:
            left = -self.innerRadius
            right = self.innerRadius
            linestyle = '--'
            locax.plot([left, right], [self.hmax, self.hmax], color = 'black', linestyle = linestyle)
            locax.plot([left, right], [self.hmin, self.hmin], color = 'black', linestyle = linestyle)
            locax.plot([left, left], [self.hmin, self.hmax], color = 'black', linestyle = linestyle)
            locax.plot([right, right], [self.hmin, self.hmax], color = 'black', linestyle = linestyle)
        plt.xlim(xmin, xmax)
        plt.ylim(zmin, zmax)
        plt.show(block = True)
        

    def plt_horizontal_Tslice(self, xmin, xmax, ymin, ymax, z, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, ny = 200, log = False, v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False):
        """
        Plot potential in a horizontal slice (normal to cylinder main axis)
        """
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
                
            Txx, Txy, Txz, Tyy, Tyz, Tzz = self.plt_TSlice('(x,y)', xmin, xmax, ymin, ymax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, Tcomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = (nx, ny), orthogonalCross = z, log = log, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, Txx = Txx, Txy = Txy, Txz = Txz, Tyy = Tyy, Tyz = Tyz, Tzz = Tzz)
            trace = Txx + Tyy + Tzz

            theta = np.linspace(0, 2*np.pi, 500)
            xc = self.radius * np.cos(theta)
            yc = self.radius * np.sin(theta)
            if z - self.z0 <= self.hmax and z - self.z0 >= self.hmin:
                linestyle = '--'
            else:
                linestyle = ':'
            locax.plot(xc, yc, color = 'black', linestyle = linestyle)
            if self.hollow:
                xc = self.innerRadius * np.cos(theta)
                yc = self.innerRadius * np.sin(theta)
                if z - self.z0 <= self.hmax and z - self.z0 >= self.hmin:
                    linestyle = '--'
                else:
                    linestyle = ':'
                plt.plot(xc, yc, color = 'black', linestyle = linestyle)
            locax.set_xlim(xmin, xmax)
            locax.set_ylim(ymin, ymax)
            locax.set_ylabel('y [m]')
            if iplot in [1,5,9]:
                locax.set_xlabel('x [m]')
            else:
                locax.set_xticklabels([])
            iplot += 1

        #trace...
        #TO DO...

        #... and conclude
        #plt.suptitle(r'$a(' + laborth+ ', z)$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        plt.show(block = True)

    def plt_vertical_Tslice(self, xmin, xmax, zmin, zmax, y, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, objectRef = True, nx = 200, nz = 200, log = False, v_method = 'regular grid', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, memoryFriendly = False, verbose = False):
        """
        Plot potential in a horizontal slice (normal to cylinder main axis)
        """
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

            Txx, Txy, Txz, Tyy, Tyz, Tzz = self.plt_TSlice('(x,z)', xmin, xmax, zmin, zmax, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, acomp = comp[iplot-1], pltAxis = locax, objectRef = objectRef, n = (nx, nz), orthogonalCross = y, log = log, v_method = v_method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = memoryFriendly, verbose = verbose, hold = True, Txx = Txx, Txy = Txy, Txz = Txz, Tyy = Tyy, Tyz = Tyz, Tzz = Tzz)
            if y - self.y0 <= self.radius and y - self.y0 >= -self.radius:
                left = -np.sqrt(self.radius**2 - y**2)
                right = np.sqrt(self.radius**2 - y**2)
                linestyle = '--'
            else:
                left = -self.radius
                right = self.radius
                linestyle = ':'
            locax.plot([left, right], [self.hmax, self.hmax], color = 'black', linestyle = linestyle)
            locax.plot([left, right], [self.hmin, self.hmin], color = 'black', linestyle = linestyle)
            locax.plot([left, left], [self.hmin, self.hmax], color = 'black', linestyle = linestyle)
            locax.plot([right, right], [self.hmin, self.hmax], color = 'black', linestyle = linestyle)
            locax.set_xlim(xmin, xmax)
            locax.set_ylim(zmin, zmax)
            locax.set_ylabel('z [m]')
            if iplot in [1,5,9]:
                locax.set_xlabel('x [m]')
            else:
                locax.set_xticklabels([])
            iplot += 1
        #plt.suptitle(r'$a(' + laborth+ ', y)$ [$m.s^{-2}$]', fontsize = 13)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        plt.show(block = True)
        
    def plt_horizontal_rhoslice(self, xmin, xmax, ymin, ymax, z, objectRef = True, nx = 200, ny = 200, log = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = False):
        """
        Plot potential in a horizontal slice (normal to cylinder main axis)
        """
        self.plt_rhoSlice('(x,y)', xmin, xmax, ymin, ymax, n = (nx, ny), orthogonalCross = z, hold = True)
        theta = np.linspace(0, 2*np.pi, 500)
        xc = self.radius * np.cos(theta)
        yc = self.radius * np.sin(theta)
        if z - self.z0 <= self.hmax and z - self.z0 >= self.hmin:
            linestyle = '--'
        else:
            linestyle = ':'
        plt.plot(xc, yc, color = 'black', linestyle = linestyle)
        if self.hollow:
            xc = self.innerRadius * np.cos(theta)
            yc = self.innerRadius * np.sin(theta)
            if z - self.z0 <= self.hmax and z - self.z0 >= self.hmin:
                linestyle = '--'
            else:
                linestyle = ':'
            plt.plot(xc, yc, color = 'black', linestyle = linestyle)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.show(block = True)


def k2p(p, a, b, L):
    n_indp = np.linspace(0, p, p+1)
    #num1 = (-1)**n_indp * factorial(2*p) * L**(-2 * n_indp)
    num1 = (-1)**n_indp * factorial(2*p)#* L**(-2 * n_indp)
    denom1 = 2**(2*n_indp) * factorial(n_indp) * factorial(n_indp+1) * factorial(2 * (p - n_indp) + 1)
    #num2 = b**(2*n_indp+2) - a**(2*n_indp+2)
    #denom2 = b**2 - a**2
    num2 = (b/L)**(2*n_indp) * (1 - (a/b)**(2*n_indp+2))
    denom2 = 1 - (a/b)**2
    #print('k2p...', p)
    #print(n_indp)
    #print(num1)
    #print(denom1)
    #print(num2)
    #print(denom2)
    #print('-------')
    return np.sum(num1 / denom1 * num2 / denom2)
        

#############################################
def tst1():
    radius = 0.5
    height = 1.6
    density = 1
    c = cylinder(radius, height, density = density)
    for i in range(4):
            for j in range(4):
                for k in range(4):
                    c.inertiaMoment(i,j,k, dx_rel = 0.005, dy_rel = 0.005, dz_rel = 0.01, verbose = False, fullNumeric = False)
                    print(i,j,k, getattr(c, 'I' + str(i) + str(j) + str(k)))
    return
    #print c.cmpVolume()
    #U = c.cmpPotential_XYZ((1.3, 2.4, 4.6), dx = 0.17, dy = 0.31, dz = 3.8)
    #U = c.cmpPotential_XYZ((1.3, 2.4, 4.6), dx = 0.003, dy = 0.002, dz = 0.05, memoryFriendly = True)
    #print U
    dx = 0.006
    dy = 0.006
    dz = 0.005
    memf = False
    timeit = True
    mc_npoints = 1e6
    im_method = 'simps'
    #im_method = 'tplquad'
    U = c.cmpPotential_XYZ((1.3, 2.4, 4.6), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit, method = 'regular grid', im_method = im_method)
    Ui = c.cmpPotential_XYZ((1.3, 2.4, 4.6), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit, method = 'inertia moments', im_method = im_method)
    Utpl = c.cmpPotential_XYZ((1.3, 2.4, 4.6), timeit = timeit, method = 'tplquad', im_method = im_method)
    #Umc = c.cmpPotential_XYZ((1.3, 2.4, 4.6), timeit = timeit, method = 'Monte Carlo', im_method = 'simps')
    #U = c.cmpPotential_XYZ_integreggrid((1.3, 2.4, 4.6), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    #Ui = c.cmpPotential_XYZ_inertiaMoments((1.3, 2.4, 4.6), dx = dx, dy = dy, dz = dz, im_method = 'tplquad', memoryFriendly = memf, timeit = timeit)
    #Utpl = c.cmpPotential_XYZ_tplquad((1.3, 2.4, 4.6), timeit = timeit)
    #Umc = c.cmpPotential_XYZ_mcquad((1.3, 2.4, 4.6), npoints = mc_npoints, timeit = timeit)
    print("--->")
    print("grid:", U)
    print("inertia moment (grid): ", Ui)
    print("tplquad: ", Utpl)
    #print("MC: ", Umc)
    if not timeit:
        print("grid / inertia:", U / Ui)
        print("grid / tplquad:", U / Utpl)
        #print("grid / MC:", U / Umc)
        print("tplquad / inertia", Utpl / Ui)
    U = c.cmpPotential_XYZ_integreggrid((13, 24, 46), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Ui = c.cmpPotential_XYZ_inertiaMoments((13, 24, 46), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Utpl = c.cmpPotential_XYZ_tplquad((13, 24, 46), timeit = timeit)
    #Umc = c.cmpPotential_XYZ_mcquad((13, 24, 46), npoints = mc_npoints, timeit = timeit)
    print("--->")
    print("grid:", U)
    print("inertia moment (grid): ", Ui)
    print("tplquad: ", Utpl)
    #print("MC: ", Umc)
    if not timeit:
        print("grid / inertia:", U / Ui)
        print("grid / tplquad:", U / Utpl)
        #print("grid / MC:", U / Umc)
        print("tplquad / inertia", Utpl / Ui)
    U = c.cmpPotential_XYZ_integreggrid((130, 240, 460), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Ui = c.cmpPotential_XYZ_inertiaMoments((130, 240, 460), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Utpl = c.cmpPotential_XYZ_tplquad((130, 240, 460), timeit = timeit)
    #Umc = c.cmpPotential_XYZ_mcquad((130, 240, 460), npoints = mc_npoints, timeit = timeit)
    print("--->")
    print("grid:", U)
    print("inertia moment (grid): ", Ui)
    print("tplquad: ", Utpl)
    #print("MC: ", Umc)
    if not timeit:
        print("grid / inertia:", U / Ui)
        print("grid / tplquad:", U / Utpl)
        #print("grid / MC:", U / Umc)
        print("tplquad / inertia", Utpl / Ui)
    U = c.cmpPotential_XYZ_integreggrid((0.02, 0, 460), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Ui = c.cmpPotential_XYZ_inertiaMoments((0.02, 0, 460), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Utpl = c.cmpPotential_XYZ_tplquad((0.02, 0, 460), timeit = timeit)
    #Umc = c.cmpPotential_XYZ_mcquad((0.02, 0, 460), npoints = mc_npoints, timeit = timeit)
    print("--->")
    print("grid:", U)
    print("inertia moment (grid): ", Ui)
    print("tplquad: ", Utpl)
    #print("MC: ", Umc)
    if not timeit:
        print("grid / inertia:", U / Ui)
        print("grid / tplquad:", U / Utpl)
        #print("grid / MC:", U / Umc)
        print("tplquad / inertia", Utpl / Ui)
    U = c.cmpPotential_XYZ_integreggrid((0.02, 0, 2), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Ui = c.cmpPotential_XYZ_inertiaMoments((0.02, 0, 2), dx = dx, dy = dy, dz = dz, memoryFriendly = memf, timeit = timeit)
    Utpl = c.cmpPotential_XYZ_tplquad((0.02, 0, 2), timeit = timeit)
    #Umc = c.cmpPotential_XYZ_mcquad((0.02, 0, 2), npoints = mc_npoints, timeit = timeit)
    print("--->")
    print("grid:", U)
    print("inertia moment (grid): ", Ui)
    print("tplquad: ", Utpl)
    #print("MC: ", Umc)
    if not timeit:
        print("grid / inertia:", U / Ui)
        print("grid / tplquad:", U / Utpl)
        #print("grid / MC:", U / Umc)
        print("tplquad / inertia", Utpl / Ui)
    #I = c.inertiaMoment(0, 1, 0, dx = 0.003, dy = 0.002, dz = 0.05, memoryFriendly = True)
    #print I
    #I = c.inertiaMoment(0, 1, 0, dx = 0.003, dy = 0.002, dz = 0.05, memoryFriendly = False)
    #print I
    #I = c.inertiaMoment(1, 1, 2, dx = 0.003, dy = 0.002, dz = 0.05, memoryFriendly = True)
    #print I
    #I = c.inertiaMoment(1, 1, 2, dx = 0.003, dy = 0.002, dz = 0.05, memoryFriendly = False)
    #print I
    I = c.inertiaMoment(1, 1, 1, dx = 0.003, dy = 0.002, dz = 0.05, memoryFriendly = False)
    print(I)
    return
    #U = c.cmpPotential_XYZ_integreggrid((-1.3, 2.4, 14.6), dx = None, dy = None, dz = None)
    #print U
    #c.plt_vr(0, 2, nr = 20)


def cmpMass(dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01):
    radius = 300 #m
    #innerRadius = 130 #m
    innerRadius = 0 #m
    height = 500 #m
    #density = 2830 #kg/m^3
    density = 1 #kg/m^3
    c = cylinder(radius, height, density = density, originType = 'centered', innerRadius = innerRadius)
    anaM = c.cmpMass()
    numM_cart = c.cmpMassNumeric(gridType = 'Cartesian', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel)
    numM_polar = c.cmpMassNumeric(gridType = 'polar', dr_rel = dr_rel, dtheta_rel = dtheta_rel)
    print('Mass -- ana, num/cart, num/polar')
    print(anaM, numM_cart, numM_polar)
    print('Ratio to ana -- cart, polar')
    print(numM_cart / anaM, numM_polar / anaM)
        

def pltVr(z = None, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, log = True, nr = 100, compareWAna = False, pmax = 15, gridType = 'Cartesian'):
    if (compareWAna or method == 'ana multipolar') and originType == 'low':
        print("To compare with analytic, need originType = centered")
        originType = 'centered'
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 300 #m
    #innerRadius = 130 #m
    innerRadius = 0 #m
    height = 500 #m
    density = 2830 #kg/m^3
    if z is None:
        if originType == 'low':
            z = height / 2
        else:
            z = 0
    c = cylinder(radius, height, density = density, originType = originType, innerRadius = innerRadius)
    c.plt_vr(-4000, 4000, nr = nr, z = z, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, compareWAna = compareWAna, pmax = pmax, gridType = gridType)
    #c.plt_vr(0, 400, nr = nr, z = z, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, compareWAna = compareWAna, pmax = pmax)
    #c.plt_vr(280, 800, nr = nr, z = 0.8, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)


def pltVz(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True, nz = 120):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_vz(280, 800, nz = nz, method = method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

    
def pltVh_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, ny = 15, log = True):
    #radius = 1.1
    #innerRadius = 0#.8
    #height = 1.6
    #density = 1
    #xmin = -2.
    #xmax = 2
    #ymin = -2
    #ymax = 2
    ##xmin = -2.5
    ##xmax = 4
    ##ymin = -3
    ##ymax = 5
    radius = 300 #m
    innerRadius = 0
    height = 500 #m
    density = 2830 #kg/m^3
    xmin = -1300
    xmax = 1300
    ymin = -1300
    ymax = 1300
    im_method = 'simps'
    #im_method = 'tplquad'
    c = cylinder(radius, height, density = density, originType = originType, innerRadius = innerRadius)
    c.plt_horizontal_vslice(xmin, xmax, ymin, ymax, height / 2, objectRef = True, nx = nx, ny = ny, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_horizontal_vslice(xmin, xmax, ymin, ymax, 2*height, objectRef = True, nx = nx, ny = ny, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltVv_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, nz = 15, log = True):
    radius = 0.8
    height = 1.6
    density = 1
    xmin = -2.5
    xmax = 4
    zmin = -3
    zmax = 5
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_vertical_vslice(xmin, xmax, zmin, zmax, 0, objectRef = True, nx = nx, nz = nz, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_vertical_vslice(xmin, xmax, zmin, zmax, 0.5*radius, objectRef = True, nx = nx, nz = nz, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_vertical_vslice(xmin, xmax, zmin, zmax, 1.3*radius, objectRef = True, nx = nx, nz = nz, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)


def pltAr(z = None, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, pmax = 15, compareWAna = False, log = True, nr = 100, gridType = 'Cartesian'):
    if (compareWAna or method == 'ana multipolar') and originType == 'low':
        print("To compare with analytic, need originType = centered")
        originType = 'centered'
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 300 #m
    innerRadius = 200 #m
    height = 5 #m
    density = 2830 #kg/m^3
    if z is None:
        if originType == 'low':
            z = height / 2
        else:
            z = 0
    c = cylinder(radius, height, density = density, originType = originType, innerRadius = innerRadius)
    c.plt_ar(280, 2800, nr = nr, z = z, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, compareWAna = compareWAna, pmax = pmax, gridType = gridType)
    #c.plt_ar(280, 800, nr = nr, z = 2 * height, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltAz(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True, nz = 120):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_az(400, 1000, nz = nz, method = method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_az(400, 1000, nz = nz, xcross = 1.3, ycross = 0, yscale = yscale, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltAh_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, ny = 15, log = True, innerRadius = 0):
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    xmin = -5500
    xmax = 5500
    ymin = -5500
    ymax = 5500
    c = cylinder(radius, height, density = density, originType = originType, innerRadius = innerRadius)
    c.plt_horizontal_aslice(xmin, xmax, ymin, ymax, height / 2, objectRef = True, nx = nx, ny = ny, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_horizontal_aslice(xmin, xmax, ymin, ymax, 2*height, objectRef = True, nx = nx, ny = ny, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltAv_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, nz = 15, log = True):
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    xmin = -800
    xmax = 800
    zmin = -1100
    zmax = 1100
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_vertical_aslice(xmin, xmax, zmin, zmax, 0, objectRef = True, nx = nx, nz = nz, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_vertical_aslice(xmin, xmax, zmin, zmax, 1.2*radius, objectRef = True, nx = nx, nz = nz, log = log, method = method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = 1e6, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltTr(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True, nr = 100):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_Tr(300, 800, nr = nr, z = height / 2, yscale = yscale, v_method = v_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)#, fname = 'Figure_Tr1.png')
    c.plt_Tr(300, 800, nr = nr, z = height, yscale = yscale, v_method = v_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)#, fname = 'Figure_Tr2.png')

    
def pltTz(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, log = True, nz = 120):
    if log:
        yscale = 'log'
    else:
        yscale = 'linear'
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_Tz(400, 1000, nz = nz, v_method = v_method, yscale = yscale, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_Tz(400, 1000, nz = nz, xcross = 1.3, ycross = 0, yscale = yscale, v_method = v_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltTh_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, ny = 15, log = True):
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    xmin = -1300
    xmax = 1300
    ymin = -1300
    ymax = 1300
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_horizontal_Tslice(xmin, xmax, ymin, ymax, height / 2, objectRef = True, nx = nx, ny = ny, log = log, v_method = v_method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_horizontal_Tslice(xmin, xmax, ymin, ymax, 2*height, objectRef = True, nx = nx, ny = ny, log = log, v_method = v_method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

def pltTv_slice(alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, v_method = 'regular grid', originType = 'low', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, nx = 15, nz = 15, log = True):
    radius = 300 #m
    height = 500 #m
    density = 2830 #kg/m^3
    xmin = -800
    xmax = 800
    zmin = -1100
    zmax = 1100
    c = cylinder(radius, height, density = density, originType = originType)
    c.plt_vertical_Tslice(xmin, xmax, zmin, zmax, 0, objectRef = True, nx = nx, nz = nz, log = log, v_method = v_method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)
    c.plt_vertical_Tslice(xmin, xmax, zmin, zmax, 1.2*radius, objectRef = True, nx = nx, nz = nz, log = log, v_method = v_method, im_method = 'simps', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, memoryFriendly = False, verbose = False, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa)

    
def pltRhoh_slice(nx = 15, ny = 15):
    radius = 300 #m
    innerRadius = 200 #m
    height = 500 #m
    density = 2830 #kg/m^3
    xmin = -350
    xmax = 350
    ymin = -350
    ymax = 350
    c = cylinder(radius, height, density = density, originType = 'low', innerRadius = innerRadius)
    c.plt_horizontal_rhoslice(xmin, xmax, ymin, ymax, height / 2, nx = nx, ny = ny)
 
