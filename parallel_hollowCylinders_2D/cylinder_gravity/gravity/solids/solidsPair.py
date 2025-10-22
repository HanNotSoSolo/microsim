import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from gravity.solids import cylinder, ball
from scipy.constants import G
from scipy.special import ellipk, ellipe
from scipy.integrate import quad, nquad, dblquad
from gravity.solids.progressETA import progress
import time
#from gravity.solids.analCyls import anaIxz, numIxz
from gravity.solids.anaCGrav import anaIxz, anaIzx, numIxz, numIzx
from gravity.solids.anaCGrav2 import cmpFz, cmpFr, cmpStiffness, cmpForceTaylor3
from gravity.solids import plots
import mpmath

class solids2(object):
    def __init__(self, obj1, obj2, resolution = {'dx_rel': 0.01, 'dy_rel': 0.01, 'dz_rel': 0.01, 'dr_rel': 0.01, 'dtheta_rel': 0.01}):
        self.obj1 = obj1
        self.obj2 = obj2
        if 'dx_rel' in resolution:
            self.dx_rel = resolution['dx_rel']
        else:
            print("WARNING! No dx_rel in resolution dictionary. Using default: 0.01")
            self.dx_rel = 0.01
        if 'dy_rel' in resolution:
            self.dy_rel = resolution['dy_rel']
        else:
            print("WARNING! No dy_rel in resolution dictionary. Using default: 0.01")
            self.dy_rel = 0.01
        if 'dz_rel' in resolution:
            self.dz_rel = resolution['dz_rel']
        else:
            print("WARNING! No dz_rel in resolution dictionary. Using default: 0.01")
            self.dz_rel = 0.01
        if 'dr_rel' in resolution:
            self.dr_rel = resolution['dr_rel']
        else:
            print("WARNING! No dr_rel in resolution dictionary. Using default: 0.01")
            self.dr_rel = 0.01
        if 'dtheta_rel' in resolution:
            self.dtheta_rel = resolution['dtheta_rel']
        else:
            print("WARNING! No dtheta_rel in resolution dictionary. Using default: 0.01")
            self.dtheta_rel = 0.01


    def getCartesianResolution(self, dx_rel, dy_rel, dz_rel):
        if dx_rel is None:
            dx_rel = self.dx_rel
        if dy_rel is None:
            dy_rel = self.dy_rel
        if dz_rel is None:
            dz_rel = self.dz_rel
        return dx_rel, dy_rel, dz_rel

    def getCylindricalResolution(self, dr_rel, dtheta_rel, dz_rel):
        if dr_rel is None:
            dr_rel = self.dr_rel
        if dtheta_rel is None:
            dtheta_rel = self.dtheta_rel
        if dz_rel is None:
            dz_rel = self.dz_rel
        return dr_rel, dtheta_rel, dz_rel
        
            
    def checkGeometry(self, pos1, pos2, dx_rel = None, dy_rel = None, dz_rel = None, raisePositionError = True, fast = True, verbose = False):
        """
        Check that objects don't overlap
        """
        #raisePositionError = True
        dx_rel, dy_rel, dz_rel = self.getCartesianResolution(dx_rel, dy_rel, dz_rel)
        self.obj1.setOrigin(pos1)
        self.obj2.setOrigin(pos2)
        okay = True
        #fast and low-memory test for special cases
        done = False
        if fast:
            if self.obj1.kind == 'ball' and self.obj2.kind == 'ball':
                okay = checkBallPairGeometry(self.obj1, self.obj2, verbose = verbose)
                done = True
            elif self.obj1.kind == 'cylinder' and self.obj2.kind == 'cylinder':
                okay = checkCylinderPairGeometry(self.obj1, self.obj2, verbose = verbose)
                done = True
            elif self.obj1.kind == 'cylinder' and self.obj2.kind == 'ball' or self.obj1.kind == 'ball' and self.obj2.kind == 'cylinder':
                if self.obj1.kind == 'cylinder':
                    okay = checkCylinderBallGeometry(self.obj2, self.obj1, verbose = verbose)
                    done = True
                else:
                    okay = checkCylinderBallGeometry(self.obj1, self.obj2, verbose = verbose)
                    done = True
            else:
                pass
        if done:
            if not okay:
                if raisePositionError:
                    raise ValueError("Objects overlap!")
                else:
                    print("WARNING! Objects overlap!")
            return okay


        x1, xl1, xr1, nx, y1, yl1, yr1, ny, z1, zl1, zr1, nz = self.obj1.mkGrid(dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, borders = True, _type = 'Cartesian')
        x2, xl2, xr2, nx, y2, yl2, yr2, ny, z2, zl2, zr2, nz = self.obj2.mkGrid(dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, borders = True, _type = 'Cartesian')
        xmin = min(np.min(xl1), np.min(xl2))
        xmax = max(np.max(xr1), np.max(xr2))
        ymin = min(np.min(yl1), np.min(yl2))
        ymax = max(np.max(yr1), np.max(yr2))
        zmin = min(np.min(zl1), np.min(zl2))
        zmax = max(np.max(zr1), np.max(zr2))
        x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = self.obj1.mkGrid(dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, zmin = zmin, zmax = zmax, borders = True, _type = 'Cartesian') #this grid embeds both objects
        yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
        yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
        in1 = self.obj1.inside((xl3, yl3, zl3), (xr3, yr3, zr3), gridType = 'Cartesian')
        in2 = self.obj2.inside((xl3, yl3, zl3), (xr3, yr3, zr3), gridType = 'Cartesian')
        if np.max(in1 + in2) > 1:
            okay = False
            if raisePositionError:
                raise ValueError("Objects overlap!")
            else:
                print("WARNING! Objects overlap!")
        return okay
        

    def cmpForce(self, pos1, pos2, _dir = '1->2', alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = None, dy_rel = None, dr_rel = None, dtheta_rel = None, dz_rel = None, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, verbose = False, raisePositionError = False, fast = False, infiniteCylinder = False, source_gridType = 'Cartesian', target_gridType = 'Cartesian', plotTGrid = False, plotAField = False, xeq = 0, yeq = 0, zeq = 0):
        """
        Compute force between balls positioned at o1 and o2
        target_gridType = 'polar' only if not fast
        """
        dx_rel, dy_rel, dz_rel = self.getCartesianResolution(dx_rel, dy_rel, dz_rel)
        dr_rel, dtheta_rel, dz_rel = self.getCylindricalResolution(dr_rel, dtheta_rel, dz_rel)
        if np.shape(pos1) != np.shape(pos2):
            raise TypeError("Conflicting positions! Must be lists of 3-elements arrays, with the same shape")
        nr = np.shape(pos1)[0]
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
            pos_s = pos1
            pos_t = pos2
            #xeq1 = 0
            #yeq1 = 0
            #zeq1 = 0
            #xeq2 = xeq
            #yeq2 = yeq
            #zeq2 = zeq
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
            pos_s = pos2
            pos_t = pos1
            #xeq1 = xeq
            #yeq1 = yeq
            #zeq1 = zeq
            #xeq2 = 0
            #yeq2 = 0
            #zeq2 = 0
        else:
            raise ValueError("Bad _dir!")

        #print(pos_t)
        #
        #for i in range(nr):
        #    pos_t[i] = list(np.array(pos_t[i]) + np.array([xeq, yeq, zeq]))
        #
        #print(pos_t)
        #raise NotImplementedError('gah!!!')
            
        #some variables used for sanity checks and plots
        if nr > 1:
            xpos = np.transpose(pos_t)[0]
            ypos = np.transpose(pos_t)[1]
            zpos = np.transpose(pos_t)[2]
            deltax_pos = np.min(np.abs(np.diff(xpos)))
            deltaz_pos = np.min(np.abs(np.diff(zpos)))
            min_xoffset = np.min(xpos)
            max_xoffset = np.max(xpos)
            min_yoffset = np.min(ypos)
            max_yoffset = np.max(ypos)
            min_zoffset = np.min(zpos)
            max_zoffset = np.max(zpos)
        else:
            min_xoffset = pos_t[0]
            max_xoffset = pos_t[0]
            min_yoffset = pos_t[1]
            max_yoffset = pos_t[1]
            min_zoffset = pos_t[2]
            max_zoffset = pos_t[2]

        if fast and not source.kind in ['cylinder', 'ball']:
            print("WARNING! gravity.solidsPair.cmpForce: fast (using cylindrical symmetry about z) available for ball and cylinder source only")
            fast = False
        if fast and target_gridType != 'Cartesian':
            print("WARNING! gravity.solidsPair.cmpForce: fast accomodates only Cartesian grid for target")
            target_gridType = 'Cartesian'
        if infiniteCylinder and source.kind != 'cylinder':
            infiniteCylinder = False
        min_roffset = np.sqrt(min_xoffset**2 + min_yoffset**2)
        if min_roffset < source.innerRadius:
            nestedCylinders = True
        else:
            nestedCylinders = False

        if source.x0 != 0 or source.y0 != 0 or source.z0 != 0:
            # For implementation purposes (grid definition below), the source should be at the origine; if it's not the case, then we just perform a translation of the system to put the source on (0,0,0)
            #print(source.x0, source.y0, source.z0)
            #print(target.x0, target.y0, target.z0)
            translated = True
            trans_x = -source.x0
            trans_y = -source.y0
            trans_z = -source.z0
            source.x0 += trans_x
            source.y0 += trans_y
            source.z0 += trans_z
            target.x0 += trans_x
            target.y0 += trans_y
            target.z0 += trans_z
            #print(source.x0, source.y0, source.z0)
            #print(target.x0, target.y0, target.z0)
            print("solidsPair.cmpForce: translating the system by (" + str(trans_x) + ", " + str(trans_y) + ", " + str(trans_z) + ") to put source at the origin")
            #raise NotImplementedError("Source must be at the origin! (not sure grid is well centered...)")
        else:
            translated = False
        
        #compute gravity field created by source on cube embedding all positions of the target
        dx_rel_target = dx_rel
        dy_rel_target = dy_rel
        dz_rel_target = dz_rel
        if not fast:
            #in this case, we compute acceleration on the entire 3D grid enclosing the positions of the target
            #if we're dealing with a cylinder and gridType == 'polar', xmin/max is actually rmin/max and ymin/max thetamin/max
            xmin_target, xmax_target, ymin_target, ymax_target, zmin_target, zmax_target = self.setLimits4cmpForce(pos_t, _dir, target_gridType)
            x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = target.mkGrid(dx_rel = dx_rel_target, dy_rel = dy_rel_target, dz_rel = dz_rel_target, xmin = xmin_target, xmax = xmax_target, ymin = ymin_target, ymax = ymax_target, zmin = zmin_target, zmax = zmax_target, borders = True, _type = target_gridType)
        else:
            #in this case, we compute acceleration only on the (x,z) plane, so we have only one value for y
            xmin_target, xmax_target, ymin_target, ymax_target, zmin_target, zmax_target, xmin_target_saved, xmax_target_saved, ymin_target_saved, ymax_target_saved, zmin_target_saved, zmax_target_saved = self.setLimits4cmpFastForce(pos_t, _dir, infiniteCylinder = infiniteCylinder, nestedCylinders = nestedCylinders)
            x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = target.mkGrid(dx_rel = dx_rel_target, dy_rel = dy_rel_target, dz_rel = dz_rel_target, xmin = xmin_target, xmax = xmax_target, ymin = ymin_target, ymax = ymax_target, zmin = zmin_target, zmax = zmax_target, borders = True, _type = target_gridType)

        #check that grid is not too coarse compared to the successive positions of the target
        if nr > 1:
            dx = x[1] - x[0]
            dz = z[1] - z[0]
            if dx > deltax_pos and deltax_pos != 0:
                print('WARNING! solidsPair.cmpForce: the target grid is too coarse compared to the successive positions of the target (x-size of the grid = ' + str(dx) + ', minimal offset along x = ' + str(deltax_pos) + '. The force will not evolve monotonically!')
            if dz > deltaz_pos and deltaz_pos != 0:
                print('WARNING! solidsPair.cmpForce: the target grid is too coarse compared to the successive positions of the target (z-size of the grid = ' + str(dz) + ', minimal offset along z = ' + str(deltaz_pos) + '. The force will not evolve monotonically!')

        #show the grid on the target
        if plotTGrid:
            plt_targetGrid(x, y, z, nx, ny, nz, min_xoffset, max_xoffset, min_yoffset, max_yoffset, min_zoffset, max_zoffset, target, source, target_gridType)

            
        #actually compute acceleration on the grid
        xag, yag, zag, ax, ay, az = source.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = (nx * ny * nz > 1e7), pmax = pmax, verbose = verbose, gridType = source_gridType)
        #xag, yag, zag, ax, ay, az = source.a_grid(x, y, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, gridType = source_gridType)
        
        #print('target grid; x, y, z')
        #print(xmin_target, xmax_target)
        #print(ymin_target, ymax_target, ymin_target_saved, ymax_target_saved)
        #print(zmin_target, zmax_target, zmin_target_saved, zmax_target_saved)
        #print()
        #print(np.min(xl), np.max(xr))
        #print(np.min(yl), np.max(yr))
        #print(np.min(zl), np.max(zr))
        #print('-------')

        #and now, compute force for each position of the target
        #but before, check that there won't be a memory problem
        ref = 0.1
        memoryFriendly = False
        #if nx * ny * nz / ref**3 > 1e7:
        if 1./(dx_rel * dy_rel * dz_rel * ref**3) > 5e8:
            print("---> solidsPair.cmpForce: turning on memoryFriendly")
            memoryFriendly = True
        strictBorder = True

        #if fast and not memoryFriendly:
        if fast:
            #in the fast case, first need to rotate the acceleration computed in the (x,z) plane (or just along x-axis if ignoring edge effects, y=z=0) before summing
            #(re)compute the full 3D grid (only one on the (x,z) plane was defined and used above)
            #NB: in this case, the target grid is always Cartesian (so we can safely just sum over cells to get total force)
            x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = target.mkGrid(dx_rel = ref * dx_rel_target, dy_rel = ref * dy_rel_target, dz_rel = dz_rel_target, xmin = xmin_target_saved, xmax = xmax_target_saved, ymin = ymin_target_saved, ymax = ymax_target_saved, zmin = zmin_target_saved, zmax = zmax_target_saved, borders = True, _type = 'Cartesian')
            
        Fx = np.zeros(nr)
        Fy = np.zeros(nr)
        Fz = np.zeros(nr)
        nsteps = nr
        nth_step = 0
        j_message = 0
        last_message_time = time.time()
        time_start = time.time()
        for i in range(nr):
            nth_step += 1
            j_message, last_message_time = progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = j_message, message_period = 120, origin = "solidsPair.cmpForce (looping on target position)")

            #o1 = pos_s[i]
            #o2 = pos_t[i]
            okay = self.checkGeometry(pos_s[i], pos_t[i], dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, raisePositionError = raisePositionError, fast = True)
            if not okay:
                continue
            #self.obj1.setOrigin(o1)
            #self.obj2.setOrigin(o2)
            source.setOrigin(pos_s[i])
            target.setOrigin(pos_t[i])

            #sum acceleration (multiplied by mass) over cells that belong to the target
            if not fast:
                #in this case, it's pretty easy. Just need to sum over the cube of accelerations...
                dx = xag[1] - xag[0]
                dy = yag[1] - yag[0]
                dz = zag[1] - zag[0]
                cellV = dx * dy * dz
                cellM = target.density * cellV
                yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
                yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
                
                axp = ax
                ayp = ay
                azp = az
                in_it = target.inside((xl3, yl3, zl3), (xr3, yr3, zr3), gridType = target_gridType, strictBorder = strictBorder)
                Fx[i] = cellM * np.sum(ax * in_it)
                Fy[i] = cellM * np.sum(ay * in_it)
                Fz[i] = cellM * np.sum(az * in_it)
            else:
                #in the fast case, first need to rotate the acceleration computed in the (x,z) plane (or just along x-axis if ignoring edge effects, y=z=0) before summing
                if not memoryFriendly:
                    #(re)compute the full 3D grid (only one on the (x,z) plane was defined and used above)
                    #NB: in this case, the target grid is always Cartesian (so we can safely just sum over cells to get total force)
                    if target_gridType != 'Cartesian':
                        raise ValueError('hum, target_gridType should be Cartesian with fast implementation')
                    #ref = 0.1
                    #ref = 1
                    #x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = target.mkGrid(dx_rel = ref * dx_rel_target, dy_rel = ref * dy_rel_target, dz_rel = dz_rel_target, xmin = xmin_target_saved, xmax = xmax_target_saved, ymin = ymin_target_saved, ymax = ymax_target_saved, zmin = zmin_target_saved, zmax = zmax_target_saved, borders = True, _type = 'Cartesian')
                    if plotTGrid:
                        #no need to min_xoffset etc here because the grid is already correctly centered on the target
                        plt_targetGrid(x, y, z, nx, ny, nz, 0, 0, 0, 0, 0, 0, target, source, 'Cartesian')

                    dx = x[1] - x[0]
                    dy = y[1] - y[0]
                    dz = z[1] - z[0]
                    cellV = dx * dy * dz
                    cellM = target.density * cellV
                    y3, x3, z3 = np.meshgrid(y, x, z) #invert x and y to avoid transpose
                    yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
                    yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
                    in_it = target.inside((xl3, yl3, zl3), (xr3, yr3, zr3), gridType = target_gridType, strictBorder = strictBorder)
                    #print("  ", nx, ny, nz)
                    #print("   ", dx, dy, dz)
                    #print("      ", cellV)
                    #print("        ", np.shape(in_it), np.sum(in_it))
                    interpKind = 'cubic'
                    #interpKind = 'linear'
                    if not infiniteCylinder:
                        axp = np.zeros(np.shape(in_it))
                        ayp = np.zeros(np.shape(in_it))
                        azp = np.zeros(np.shape(in_it))
                        for iz in range(nz):
                            xp = x3[:,:,iz]
                            yp = y3[:,:,iz]
                            yp, xp = np.meshgrid(y, x)
                            rp = np.sqrt(xp**2 + yp**2)
                            thetap = np.arctan2(yp, xp)
                            #funx = interp1d(x, ax[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            #funy = interp1d(x, ay[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            #funz = interp1d(x, az[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            funx = interp1d(xag, ax[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            funy = interp1d(xag, ay[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            funz = interp1d(xag, az[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            #ax = np.cos(thetap) * funx(rp) - np.sin(thetap) * funy(rp)
                            #ay = np.sin(thetap) * funx(rp) + np.cos(thetap) * funy(rp)
                            axp[:,:,iz] = np.cos(thetap) * funx(rp) #funx(rp) is ax taken along the x-axis, i.e. the (signed) norm of the acceleration; this is exactly like the full rotation above, but with the symmetry assumption y->-y explicit
                            ayp[:,:,iz] = np.sin(thetap) * funx(rp)
                            azp[:,:,iz] = funz(rp)

                        Fx[i] = cellM * np.nansum(axp * in_it)
                        Fy[i] = cellM * np.nansum(ayp * in_it)
                        Fz[i] = cellM * np.nansum(azp * in_it)
                
                    else:
                        raise NotImplementedError("Should be obsolete... Useless and dangerous!")

                else:
                    #more leverage on memory
                    #for iy in range(ny):
                    ##x, xl, xr, nx, y_dummy, yl_dummy, yr_dummy, ny_dummy, z, zl, zr, nz = target.mkGrid(dx_rel = ref * dx_rel_target, dy_rel = ref * dy_rel_target, dz_rel = dz_rel_target, xmin = xmin_target_saved, xmax = xmax_target_saved, ymin = y[iy], ymax = y[iy], zmin = zmin_target_saved, zmax = zmax_target_saved, borders = True, _type = 'Cartesian')
                    #x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = target.mkGrid(dx_rel = ref * dx_rel_target, dy_rel = ref * dy_rel_target, dz_rel = dz_rel_target, xmin = xmin_target_saved, xmax = xmax_target_saved, ymin = ymin_target_saved, ymax = ymax_target_saved, zmin = zmin_target_saved, zmax = zmax_target_saved, borders = True, _type = 'Cartesian')
                    dx = x[1] - x[0]
                    dy = y[1] - y[0]
                    dz = z[1] - z[0]
                    cellV = dx * dy * dz
                    #print("  ", nx, ny, nz)
                    #print("   ", dx, dy, dz)
                    #print("      ", cellV)
                    cellM = target.density * cellV
                    for iy in range(ny):
                        y3, x3, z3 = np.meshgrid(y[iy], x, z) #invert x and y to avoid transpose
                        yl3, xl3, zl3 = np.meshgrid(yl[iy], xl, zl) #invert x and y to avoid transpose
                        yr3, xr3, zr3 = np.meshgrid(yr[iy], xr, zr) #invert x and y to avoid transpose
                        in_it = target.inside((xl3, yl3, zl3), (xr3, yr3, zr3), gridType = target_gridType, strictBorder = strictBorder)
                        #print("        ", y[iy], np.shape(in_it), np.sum(in_it))
                        axp = np.zeros(np.shape(in_it))
                        ayp = np.zeros(np.shape(in_it))
                        azp = np.zeros(np.shape(in_it))
                        for iz in range(nz):
                            xp = x3[:,0,iz]
                            yp = y3[:,0,iz]
                            yp, xp = np.meshgrid(y[iy], x)
                            rp = np.sqrt(xp**2 + yp**2)
                            thetap = np.arctan2(yp, xp)
                            interpKind = 'cubic'
                            funx = interp1d(xag, ax[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            funy = interp1d(xag, ay[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            funz = interp1d(xag, az[:,iz], kind = interpKind, bounds_error = False, fill_value = 0)
                            #ax = np.cos(thetap) * funx(rp) - np.sin(thetap) * funy(rp)
                            #ay = np.sin(thetap) * funx(rp) + np.cos(thetap) * funy(rp)
                            axp[:,:,iz] = np.cos(thetap) * funx(rp) #funx(rp) is ax taken along the x-axis, i.e. the (signed) norm of the acceleration; this is exactly like the full rotation above, but with the symmetry assumption y->-y explicit
                            ayp[:,:,iz] = np.sin(thetap) * funx(rp)
                            azp[:,:,iz] = funz(rp)

                        Fx[i] += cellM * np.nansum(axp * in_it)
                        Fy[i] += cellM * np.nansum(ayp * in_it)
                        Fz[i] += cellM * np.nansum(azp * in_it)
                    

            #plot acceleration field
            if plotAField and not memoryFriendly:
                plt_accelerationField(axp, ayp, azp, in_it, x, y, z, nx, ny, nz, target, source, Fx[i], Fy[i], Fz[i])
                    
        if translated:
            source.x0 -= trans_x
            source.y0 -= trans_y
            source.z0 -= trans_z
            target.x0 -= trans_x
            target.y0 -= trans_y
            target.z0 -= trans_z
            
        return Fx, Fy, Fz


    def cmpForce_BF(self, pos1, pos2, _dir = '1->2', alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, pmax = 15, verbose = False, raisePositionError = False, source_gridType = 'Cartesian', target_gridType = 'Cartesian'):
        """
        Compute force between balls positioned at o1 and o2
        """
        raise NotImplementedError("This implementation provides bias results. And it is soooooo slow that I even give up trying to understand why it's biased. Should not be used!")
        # #pos1 = [0,0,0]
        # #pos2 = [0,0,0]
        # if np.size(pos1) != 3 or np.size(pos2) != 3:
        #     raise TypeError("positions must be 3-element arrays [x,y,z]")
        # if _dir == '1->2':
        #     source = self.obj1
        #     target = self.obj2
        #     pos_t = pos2
        # elif _dir == '2->1':
        #     source = self.obj2
        #     target = self.obj1
        #     pos_t = pos1
        # else:
        #     raise ValueError("Bad _dir!")

        # okay = self.checkGeometry(pos1, pos2, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, raisePositionError = raisePositionError, fast = True)
        # self.obj1.setOrigin(pos1)
        # self.obj2.setOrigin(pos2)
        
        # if source.x0 != 0 or source.y0 != 0 or source.z0 != 0:
        #     # For implementation purposes (grid definition below), the source should be at the origine; if it's not the case, then we just perform a translation of the system to put the source on (0,0,0)
        #     translated = True
        #     trans_x = -source.x0
        #     trans_y = -source.y0
        #     trans_z = -source.z0
        #     source.x0 += trans_x
        #     source.y0 += trans_y
        #     source.z0 += trans_z
        #     target.x0 += trans_x
        #     target.y0 += trans_y
        #     target.z0 += trans_z
        #     print("solidsPair.cmpForce_BF: translating the system by (" + str(trans_x) + ", " + str(trans_y) + ", " + str(trans_z) + ") to put source at the origin")
        #     #raise NotImplementedError("Source must be at the origin! (not sure grid is well centered...)")
        # else:
        #     translated = False

        # #compute gravity field created by source on cube embedding all positions of the target
        # dx_rel_target = dx_rel
        # dy_rel_target = dy_rel
        # dz_rel_target = dz_rel
        # #brute-force computation: we compute acceleration on the entire 3D grid enclosing the positions of the target
        # #if we're dealing with a cylinder and gridType == 'polar', xmin/max is actually rmin/max and ymin/max thetamin/max
        # xmin_target, xmax_target, ymin_target, ymax_target, zmin_target, zmax_target = self.setLimits4cmpForce(pos_t, _dir, target_gridType)
        # x, xl, xr, nx, y, yl, yr, ny, z, zl, zr, nz = target.mkGrid(dx_rel = dx_rel_target, dy_rel = dy_rel_target, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel_target, xmin = xmin_target, xmax = xmax_target, ymin = ymin_target, ymax = ymax_target, zmin = zmin_target, zmax = zmax_target, borders = True, _type = target_gridType)
        # if target.kind == 'cylinder' and target_gridType == 'polar':
        #     #NB: in this case, points are all inside the target
        #     r = np.copy(x)
        #     theta = np.copy(y)
        #     th3, r3, z3 = np.meshgrid(theta, r, z)
        #     x3 = r3 * np.cos(th3)
        #     y3 = r3 * np.sin(th3)
        #     x3 += target.x0
        #     y3 += target.y0
        # else:
        #     dx = x[1] - x[0]
        #     dy = y[1] - y[0]
        #     dz = z[1] - z[0]
        #     cellV = dx * dy * dz
        #     y3, x3, z3 = np.meshgrid(y, x, z) #invert x and y to avoid transpose

        # x3 = np.reshape(x3, nx * ny * nz)
        # y3 = np.reshape(y3, nx * ny * nz)
        # z3 = np.reshape(z3, nx * ny * nz)
        # xyz = np.transpose(np.vstack((x3,y3,z3)))
        # # if target.kind == 'cylinder' and target_gridType == 'polar':
        # #     r3 = np.reshape(r3, nx * ny * nz)
        # #     th3 = np.reshape(th3, nx * ny * nz)
        # #     axp = plt.subplot(311)
        # #     plt.plot(r3)
        # #     plt.subplot(312, sharex = axp)
        # #     plt.plot(th3)
        # #     plt.subplot(313, sharex = axp)
        # #     plt.plot(z3)
        # #     plt.show()
        # # else:
        # #     axp = plt.subplot(311)
        # #     plt.plot(x3)
        # #     plt.subplot(312, sharex = axp)
        # #     plt.plot(y3)
        # #     plt.subplot(313, sharex = axp)
        # #     plt.plot(z3)
        # #     plt.show()
        # # stop
        # # plotit = False
        # # if plotit:
        # #     th4p = np.linspace(0, 2 * np.pi, 500)
        # #     ix4p = target.innerRadius * np.cos(th4p) + target.x0
        # #     iy4p = target.innerRadius * np.sin(th4p) + target.y0
        # #     ox4p = target.radius * np.cos(th4p) + target.x0
        # #     oy4p = target.radius * np.sin(th4p) + target.y0
        # #     plt.plot(x3, y3, marker = '.', linestyle = '', color = 'black')
        # #     plt.plot(ix4p, iy4p, color = 'blue')
        # #     plt.plot(ox4p, oy4p, color = 'blue')
        # #     plt.plot([target.x0], [target.y0], color = 'red', marker = 'x')

        # #     ix4p = source.innerRadius * np.cos(th4p) + source.x0
        # #     iy4p = source.innerRadius * np.sin(th4p) + source.y0
        # #     ox4p = source.radius * np.cos(th4p) + source.x0
        # #     oy4p = source.radius * np.sin(th4p) + source.y0
        # #     #plt.plot(x3, y3, marker = '.', linestyle = '', color = 'black')
        # #     plt.plot(ix4p, iy4p, color = 'green')
        # #     plt.plot(ox4p, oy4p, color = 'green')
        # #     plt.show()
        

        # #actually compute acceleration on the (Cartesian) grid
        # xag, yag, zag, ax, ay, az = source.a_grid2(xyz, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, pmax = pmax, verbose = verbose, gridType = source_gridType) #the gridType here is for grid over the source, not the grid on which we compute the gravity field
 
        # #sum acceleration (multiplied by mass) over cells that belong to the target
        # #in brute force case, it's pretty easy. Just need to sum over the cube of accelerations...
        # if target_gridType == 'Cartesian':
        #     #dx = xag[1] - xag[0]
        #     #dy = yag[1] - yag[0]
        #     #dz = zag[1] - zag[0]
        #     #cellV = dx * dy * dz
        #     cellM = target.density * cellV
        #     yl3, xl3, zl3 = np.meshgrid(yl, xl, zl) #invert x and y to avoid transpose
        #     yr3, xr3, zr3 = np.meshgrid(yr, xr, zr) #invert x and y to avoid transpose
        #     in_it = target.inside((xl3, yl3, zl3), (xr3, yr3, zr3), gridType = 'Cartesian')
        #     in_it = np.reshape(in_it, nx * ny * nz)
        #     #print("\n---> #### ", np.sum(ax), np.sum(in_it), np.sum(ax * in_it), cellM, " ####")
        #     Fx = cellM * np.sum(ax * in_it)
        #     Fy = cellM * np.sum(ay * in_it)
        #     Fz = cellM * np.sum(az * in_it)
        # elif target.kind == 'cylinder' and target_gridType == 'polar':
        #     #in this case, the grid is completely inside the target, so no need to target.inside()
        #     xag3 = np.reshape(xag, (nx, ny, nz))
        #     yag3 = np.reshape(yag, (nx, ny, nz))
        #     zag3 = np.reshape(zag, (nx, ny, nz))

        #     dax = np.reshape(ax, (nx, ny, nz))
        #     day = np.reshape(ay, (nx, ny, nz))
        #     daz = np.reshape(az, (nx, ny, nz))
        #     dax_xy = simps(dax, x = z, axis = 2)
        #     dax_x = simps(dax_xy, x = theta, axis = 1)
        #     ax = simps(r * dax_x, x = r, axis = 0)
        #     day_xy = simps(day, x = z, axis = 2)
        #     day_x = simps(day_xy, x = theta, axis = 1)
        #     ay = simps(r * day_x, x = r, axis = 0)
        #     daz_xy = simps(daz, x = z, axis = 2)
        #     daz_x = simps(daz_xy, x = theta, axis = 1)
        #     az = simps(r * daz_x, x = r, axis = 0)

        #     Fx = target.density * ax
        #     Fy = target.density * ay
        #     Fz = target.density * az
            
        # else:
        #     raise NotImplementedError()

        # if translated:
        #     source.x0 -= trans_x
        #     source.y0 -= trans_y
        #     source.z0 -= trans_z
        #     target.x0 -= trans_x
        #     target.y0 -= trans_y
        #     target.z0 -= trans_z

        # return Fx, Fy, Fz


        
    def setLimits4cmpForce(self, pos, _dir, gridType):
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
            #pos_t = pos2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
            #pos_t = pos1
        else:
            raise ValueError("Bad _dir!")
        if np.size(np.shape(pos)) > 1:
            xmin_target = None
            xmax_target = None
            ymin_target = None
            ymax_target = None
            zmin_target = None
            zmax_target = None
            nr = np.shape(pos)[0]
            for i in range(nr):
                xmin_target, xmax_target = getMinMax(pos[i][0], xmin_target, xmax_target)
                ymin_target, ymax_target = getMinMax(pos[i][1], ymin_target, ymax_target)
                zmin_target, zmax_target = getMinMax(pos[i][2], zmin_target, zmax_target)
        else:
            xmin_target = pos[0]
            xmax_target = pos[0]
            ymin_target = pos[1]
            ymax_target = pos[1]
            zmin_target = pos[2]
            zmax_target = pos[2]
        #if target.kind in ['ball', 'cylinder']:
        #    xmin_target -= 1.01 * target.radius
        #    xmax_target += 1.01 * target.radius
        #    ymin_target -= 1.01 * target.radius
        #    ymax_target += 1.01 * target.radius
        #else:
        #    raise NotImplementedError()
        if target.kind == 'ball':
            xmin_target -= 1.01 * target.radius
            xmax_target += 1.01 * target.radius
            ymin_target -= 1.01 * target.radius
            ymax_target += 1.01 * target.radius
            zmin_target -= 1.01 * target.radius
            zmax_target += 1.01 * target.radius
        elif target.kind == 'cylinder':
            if gridType == 'Cartesian':
                xmin_target -= 1.01 * target.radius
                xmax_target += 1.01 * target.radius
                ymin_target -= 1.01 * target.radius
                ymax_target += 1.01 * target.radius
            elif gridType == 'polar':
                xmin_target = target.innerRadius #actually rmin
                xmax_target = target.radius #actually rmax
                ymin_target = 0 #actually thetamin
                ymax_target = 2 * np.pi #actually thetamax
            else:
                raise ValueError('?!?')
            zmin_target -= 0.51 * target.height
            zmax_target += 0.51 * target.height
        else:
            raise NotImplementedError()
        return xmin_target, xmax_target, ymin_target, ymax_target, zmin_target, zmax_target

    def setLimits4cmpFastForce(self, pos, _dir, infiniteCylinder = False, nestedCylinders = False):
        #fast implementation: use cylindrical symmetry: compute acceleration on (x,z) plane (or just along the x-axis if ignoring edge effects, y=z=0), then rotate it to populate the entire box
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
            #pos_t = pos2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
            #pos_t = pos1
        else:
            raise ValueError("Bad _dir!")
        xmin_target, xmax_target, ymin_target, ymax_target, zmin_target, zmax_target = self.setLimits4cmpForce(pos, _dir, 'Cartesian')
        xmin_target_saved = np.copy(xmin_target)
        xmax_target_saved = np.copy(xmax_target)
        ymin_target_saved = np.copy(ymin_target)
        ymax_target_saved = np.copy(ymax_target)
        zmin_target_saved = np.copy(zmin_target)
        zmax_target_saved = np.copy(zmax_target)
        if nestedCylinders:
            xmin_target = xmin_target_saved + target.radius + target.innerRadius #xmin_target_saved + target.radius gives the center of the solid when at leftmost position, then we want to only sample from its inner radius
        xmax_target = xmax_target_saved
        ymin_target = 0
        ymax_target = 0
        if infiniteCylinder:
            if (source.hmin + source.hmax)/2 != (target.hmin + target.hmax)/2:
                raise ValueError("Can ignore edge effects only if cylinders are at the same altitude")
            if source.height <= 10 * source.radius:
                print("WARNING! Source cylinder height not very big compared to radius, ignoring edge effects may be a poor assumption")
            if target.height >= 0.2 * source.height:
                print("WARNING! Target cylinder height not small compared to source's, ignoring edge effects may be a poor assumption")
            zmin_target = 0
            zmax_target = 0
        return xmin_target, xmax_target, ymin_target, ymax_target, zmin_target, zmax_target, xmin_target_saved, xmax_target_saved, ymin_target_saved, ymax_target_saved, zmin_target_saved, zmax_target_saved
        
                
    def plt_horizontal_aslice(self, source, nradii = 2.5, nx = 30, ny = 30, z = 0, nr = 100, log = True, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = None, dy_rel = None, dz_rel = None, mc_npoints = 1e6, memoryFriendly = False, verbose = False, fast = False):
        """
        nradii -- range of plot, in number of outer cylinder radii
        """
        dx_rel, dy_rel, dz_rel = self.getCartesianResolution(dx_rel, dy_rel, dz_rel)
        if source == 1:
            obj = self.obj1
            obj2 = self.obj2
        elif source == 2:
            obj = self.obj2
            obj2 = self.obj1
        else:
            raise ValueError("Bad source")

        if obj.kind in ['ball', 'cylinder']:
            xmin = obj.x0 - nradii * obj.radius
            xmax = obj.x0 + nradii * obj.radius
            ymin = obj.y0 - nradii * obj.radius
            ymax = obj.y0 + nradii * obj.radius
        obj.plt_horizontal_aslice(xmin, xmax, ymin, ymax, z, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, nx = nx, ny = ny, nr = nr, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, object2 = obj2, fast = fast)

        
    def plt_vertical_aslice(self, source, nradii = 2.5, nheight = 1.5, nx = 30, nz = 30, y = 0, nr = 100, log = True, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = None, dy_rel = None, dz_rel = None, mc_npoints = 1e6, memoryFriendly = False, verbose = False, fast = False):
        """
        nradii -- range of plot, in number of outer cylinder radii
        """
        dx_rel, dy_rel, dz_rel = self.getCartesianResolution(dx_rel, dy_rel, dz_rel)
        if source == 1:
            obj = self.obj1
            obj2 = self.obj2
        elif source == 2:
            obj = self.obj2
            obj2 = self.obj1
        else:
            raise ValueError("Bad source")

        if obj.kind in ['ball', 'cylinder']:
            xmin = obj.x0 - nradii * obj.radius
            xmax = obj.x0 + nradii * obj.radius
        if obj.kind == 'cylinder':
            zmin = obj.hmin - 0.5 * nheight * obj.height
            zmax = obj.hmax + 0.5 * nheight * obj.height
        elif obj.kind == 'ball':
            zmin = obj.z0 - nradii * obj.radius
            zmax = obj.z0 + nradii * obj.radius
        else:
            pass
            #raise NotImplementedError()
        obj.plt_vertical_aslice(xmin, xmax, zmin, zmax, y, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, nx = nx, nz = nz, log = log, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, object2 = obj2)



class ballPair(solids2):
    def __init__(self, ball_1, ball_2):
        super(ballPair, self).__init__(ball_1, ball_2)

    def cmpForce_ana(self, pos1, pos2, _dir = '1->2'):
        if np.shape(pos1) != np.shape(pos2):
            raise TypeError("Conflicting positions! Must be lists of 3-elements arrays, with the same shape")
        nr = np.shape(pos1)[0]
        Fx = np.zeros(nr)
        Fy = np.zeros(nr)
        Fz = np.zeros(nr)
        for i in range(nr):
            o1 = pos1[i]
            o2 = pos2[i]
            Fx[i], Fy[i], Fz[i] = self.cmpForce_ana1(o1, o2, _dir = _dir, raisePositionError = False)
        return Fx, Fy, Fz

    def cmpForce_ana1(self, o1, o2, _dir = '1->2', raisePositionError = False, verbose = False):
        if np.array_equal(o1, o2):
            return 0, 0, 0
        self.obj1.setOrigin(o1)
        self.obj2.setOrigin(o2)
        r = np.sqrt((self.obj2.x0 - self.obj1.x0)**2 + (self.obj2.y0 - self.obj1.y0)**2 + (self.obj2.z0 - self.obj1.z0)**2)
        F = -G * self.obj1.cmpMass() * self.obj2.cmpMass() / r**3

        okay = checkBallPairGeometry(self.obj1, self.obj2, verbose = verbose)
        if not okay:
            if raisePositionError:
                raise ValueError("Geometry not allowed! Balls overlap.")
            else:
                if verbose:
                    print("Geometry not allowed! Balls overlap.")
                return 0,0,0

        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
        else:
            raise ValueError("Bad _dir!")

        Fx = F * (target.x0 - source.x0)
        Fy = F * (target.y0 - source.y0)
        Fz = F * (target.z0 - source.z0)
        return Fx, Fy, Fz



class cylinderPair(solids2):
    def __init__(self, cylinder_1, cylinder_2, resolution = {'dx_rel': 0.01, 'dy_rel': 0.01, 'dz_rel': 0.01, 'dr_rel': 0.01, 'dtheta_rel': 0.01}):
        super(cylinderPair, self).__init__(cylinder_1, cylinder_2, resolution = resolution)

    def cmpAnaFx(self, pos1, pos2, _dir = '1->2', yukawa = False, lmbda = None, alpha = None, method = 1, kmax = 50, plotIntgd = False, nk4plot = 500, xeq = 0, yeq = 0, zeq = 0):
        """
        Analytic expression along radial direction
        """
        return self.cmpAnaF('x', pos1, pos2, _dir = _dir, yukawa = yukawa, lmbda = lmbda, alpha = alpha, method = method, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot, xeq = xeq, yeq = yeq, zeq = zeq)

    def cmpAnaFz(self, pos1, pos2, _dir = '1->2', yukawa = False, lmbda = None, alpha = None, method = 1, kmax = 50, plotIntgd = False, nk4plot = 500, xeq = 0, yeq = 0, zeq = 0):
        """
        Analytic expression along symmetry axis
        """
        res = self.cmpAnaF('z', pos1, pos2, _dir = _dir, yukawa = yukawa, lmbda = lmbda, alpha = alpha, method = method, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot, xeq = xeq, yeq = yeq, zeq = zeq)
        if method == 1:
            #not sure why, but works...
            res *= -1
        return res

    def cmpAnaFx3(self, d, _dir = '1->2', lmbda = None, alpha = 1, kmax = 'auto', xeq = 0, yeq = 0, zeq = 0):
        """
        Compute force using 3rd order Taylor expansion
        """
        return self.cmpAnaF3(d, 'x', _dir = _dir, lmbda = lmbda, alpha = alpha, kmax = kmax, zeq = zeq)

    def cmpAnaFz3(self, d, _dir = '1->2', lmbda = None, alpha = 1, kmax = 'auto', xeq = 0, yeq = 0, zeq = 0):
        """
        Compute force using 3rd order Taylor expansion
        """
        return self.cmpAnaF3(d, 'z', _dir = _dir, lmbda = lmbda, alpha = alpha, kmax = kmax, zeq = zeq)
        
    
    def cmpAnaF(self, _axis, pos1, pos2, _dir = '1->2', yukawa = False, lmbda = None, alpha = None, method = 1, kmax = 50, plotIntgd = False, nk4plot = 500, xeq = 0, yeq = 0, zeq = 0):
        """
        method -- 1 for Hoyle, 2 for Lockerbie
        xeq, yeq, zeq -- equilibrium (reference) position of the target, so that its real position is pos + [xeq, yeq, zeq]. Required for when we deal with linearized forces.
        """
        mfun = lambda x: -np.pi/2
        Mfun = lambda x: np.pi/2

        #print(pos1)
        if _dir == '1->2':
            pos2 += np.array([xeq, yeq, zeq])
        elif _dir == '2->1':
            pos1 += np.array([xeq, yeq, zeq])
        else:
            raise ValueError('Bad!')
        #print(pos1)

        okay = self.checkGeometry(pos1, pos2, raisePositionError = True)

        if yukawa and alpha is None:
            raise TypeError("For yukawa, alpha must be set")

        if not _axis in ['x', 'z']:
            raise ValueError('_axis must be one of x or z')
        
        if np.shape(pos1) != np.shape(pos2):
            raise TypeError("Conflicting positions! Must be lists of 3-elements arrays, with the same shape")
        nr = np.shape(pos1)[0]
        if np.size(np.shape(pos1)) > 1 and nr > 1:
            raise NotImplementedError()
        self.obj1.setOrigin(pos1)
        self.obj2.setOrigin(pos2)
        if self.obj1.y0 != self.obj2.y0:
            raise NotImplementedError('Must rotate reference frame...')
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
            #pos_t = pos2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
            #pos_t = pos1
        else:
            raise ValueError("Bad _dir!")


        if method == 2 and (source.x0 != 0 or source.y0 != 0 or source.z0 != 0):
            # For implementation purposes (grid definition below), the source should be at the origine; if it's not the case, then we just perform a translation of the system to put the source on (0,0,0)
            #print(source.x0, source.y0, source.z0)
            #print(target.x0, target.y0, target.z0)
            translated = True
            trans_x = -source.x0
            trans_y = -source.y0
            trans_z = -source.z0
            source.x0 += trans_x
            source.y0 += trans_y
            source.z0 += trans_z
            source.hmin += trans_z
            source.hmax += trans_z
            target.x0 += trans_x
            target.y0 += trans_y
            target.z0 += trans_z
            target.hmin += trans_z
            target.hmax += trans_z
            #print(source.x0, source.y0, source.z0)
            #print(target.x0, target.y0, target.z0)
            print("solidsPair.cmpAnaF: translating the system by (" + str(trans_x) + ", " + str(trans_y) + ", " + str(trans_z) + ") to put source at the origin")
            #raise NotImplementedError("Source must be at the origin! (not sure grid is well centered...)")
        else:
            translated = False

        #if source.x0 != 0 or source.y0 != 0 or source.z0 != 0:
        #    raise NotImplementedError("Source should be at (0,0,0) -- (" +  str(source.x0) + ", " + str(source.y0) + ", " + str(source.z0) + ")")
        t = target.x0
        #s = target.z0 - source.z0 - 0.5 * (source.height + target.height)
        a1 = source.radius
        a2 = target.radius
        ia1 = source.innerRadius
        ia2 = target.innerRadius
        z1min = source.hmin
        z1max = source.hmax
        z2min = target.hmin
        z2max = target.hmax
        if method == 1:
            if _axis == 'x':
                res, err = dblquad(cmp_yint_x, 0, np.pi/2, mfun, Mfun, args = (a1, a2, t, z1min, z1max, z2min, z2max, ia1, ia2, yukawa, lmbda))
            else:
                res, err = dblquad(cmp_yint_z, 0, np.pi/2, mfun, Mfun, args = (a1, a2, t, z1min, z1max, z2min, z2max, ia1, ia2, yukawa, lmbda))
            F = 2 * G * source.density * target.density * res
            if yukawa:
                F *= alpha

        elif method == 2:
            if _axis == 'x':
                _fun = cmpFr
            else:
                _fun = cmpFz
            #if 0.5 * (z1min + z1max) != 0:
            #    raise ValueError('Source must be at z=0')
            if source.x0 != 0 or source.y0 != 0 or source.z0 != 0:
                raise NotImplementedError("Source should be at (0,0,0) -- (" +  str(source.x0) + ", " + str(source.y0) + ", " + str(source.z0) + ")")
            zs = 0.5 * (z2min + z2max)
            F = _fun(t, zs, a2, 0.5*target.height, target.density, ia1, a1, 0.5*source.height, source.density, alpha = alpha, lmbda = lmbda, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)
            if ia2 > 0:
                iF = _fun(t, zs, ia2, 0.5*target.height, target.density, ia1, a1, 0.5*source.height, source.density, alpha = alpha, lmbda = lmbda, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)
                F -= iF
            #if _axis == 'x':
            #    F *= np.sign(t)
        else:
            raise ValueError('Bad method', method)
        return F

    def cmpAnaF3(self, d, _axis, _dir = '1->2', lmbda = None, alpha = 1, kmax = 'auto', zeq = 0):
        """
        Compute force using 3rd order Taylor expansion
        zeq -- reference position of the target along z (the source being centered on 0). Should be equal to (target.z0 - source.z0), but I'm not entirely sure of how general this is... so keep this annoying zeq...
        """
        if not _axis in ['x', 'z']:
            raise ValueError('_axis must be one of x or z')

        if lmbda is None:
            raise ValueError('lmbda should be set... at least for now...')
        
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
        else:
            raise ValueError("Bad _dir!")

        if zeq != (target.z0 - source.z0):
            print('WARNING!!! solidsPair.cmpAnaF3: Should be a problem... Worth investigating, anyway!!! zeq != target.z0 - source.z0', zeq, target.z0 - source.z0)
            #raise ValueError('Should be a problem... Worth investigating, anyway!!! zeq != target.z0 - source.z0', zeq, target.z0 - source.z0)
        
        F = cmpForceTaylor3(d, _axis, target.innerRadius, target.radius, 0.5 * target.height, target.density, source.innerRadius, source.radius, 0.5 * source.height, source.density, zs = zeq, alpha = alpha, lmbda = lmbda, kmax = kmax)
        return F

    def cmpStiffness(self, _axis, _dir = '1->2', lmbda = None, alpha = 1, kmax = 'auto', xeq = 0, yeq = 0, zeq = 0):
        """
        method -- 1 for Hoyle, 2 for Lockerbie
        xeq, yeq, zeq -- equilibrium (reference) position of the target, so that its real position is pos + [xeq, yeq, zeq]. Required for when we deal with linearized forces.
        """
        if not _axis in ['x', 'z']:
            raise ValueError('_axis must be one of x or z')

        if lmbda is None:
            raise ValueError('lmbda should be set... at least for now...')
        
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
        else:
            raise ValueError("Bad _dir!")

        ky = cmpStiffness(_axis, target.innerRadius, target.radius, 0.5 * target.height, target.density, source.innerRadius, source.radius, 0.5 * source.height, source.density, zs = zeq, alpha = alpha, lmbda = lmbda, kmax = kmax)
        #if _axis == 'x':
        #    ky = cmpRadialStiffness(target.innerRadius, target.radius, 0.5 * target.height, target.density, source.innerRadius, source.radius, 0.5 * source.height, source.density, zs = zeq, alpha = alpha, lmbda = lmbda, kmax = kmax)
        #else:
        #    ky = cmpLongitudinalStiffness(target.innerRadius, target.radius, 0.5 * target.height, target.density, source.innerRadius, source.radius, 0.5 * source.height, source.density, zs = zeq, alpha = alpha, lmbda = lmbda, kmax = kmax)
        return ky
    
    def cmpAnaFx_JPU(self, d):
        """
        Compute analytic expression of radial force created by inner cylinder on outer one, for small d. JPU, bit incorrect
        """
        e1 = self.obj1.radius - self.obj1.innerRadius
        R1 = self.obj1.innerRadius
        e2 = self.obj2.radius - self.obj2.innerRadius
        R2 = self.obj2.innerRadius
        if e1 > 0.1 * R1:
            raise ValueError("Cylinder #1 too thick!")
        if e2 > 0.1 * R2:
            raise ValueError("Cylinder #2 too thick!")
        if self.obj1.height <= R1:
            raise ValueError("Cylinder #1 too small")
        if self.obj2.height <= R2:
            raise ValueError("Cylinder #2 too small")
        bad = np.where(np.abs(d) > 0.1 * min(R1, R2))[0]
        if np.size(bad) > 0:
            raise ValueError("Some displacements too big!")
        if R1 < R2:
            bad = np.where(np.abs(d) > self.obj2.innerRadius - self.obj1.radius)[0]
            if np.size(bad) > 0:
                raise ValueError("Geometry not allowed")
        if R2 < R1:
            bad = np.where(np.abs(d) > self.obj1.innerRadius - self.obj2.radius)[0]
            if np.size(bad) > 0:
                raise ValueError("Geometry not allowed")
        
        M1 = self.obj1.cmpMass()
        M2 = self.obj2.cmpMass()
        Delta = self.obj1.radius / self.obj2.radius
        lmbda = self.obj2.height / self.obj2.radius
        k = -4 * G * M1 * M2 / R2**3 * (cmpIx(Delta, lmbda) - Delta * cmpJx(Delta, lmbda)) #NB: in JPU's notes, \hat{delta} = delta/R2, and here d=delta, thus R2**3 instead of R2**2 in JPU's notes
        Fx = k * d
        return Fx

    def cmpEWFx(self, pos1, pos2, _dir = '1->2'):
        """
        Analytic expression from Hoyle et al (Eot-Wash)
        """
        mfun = lambda x: 0
        Mfun = lambda x: np.pi/2
        
        if self.obj1.innerRadius > 0 or self.obj2.innerRadius > 0:
            raise NotImplementedError('Hollow cylinders not ready...')
        if np.shape(pos1) != np.shape(pos2):
            raise TypeError("Conflicting positions! Must be lists of 3-elements arrays, with the same shape")
        nr = np.shape(pos1)[0]
        if np.size(np.shape(pos1)) > 1 and nr > 1:
            raise NotImplementedError()
        self.obj1.setOrigin(pos1)
        self.obj2.setOrigin(pos2)
        if self.obj1.y0 != self.obj2.y0:
            raise NotImplementedError('Must rotate reference frame...')
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
            pos_t = pos2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
            pos_t = pos1
        else:
            raise ValueError("Bad _dir!")

        if source.x0 != 0 or source.y0 != 0 or source.z0 != 0:
            raise NotImplementedError("Source should be at (0,0,0) -- (" +  str(source.x0) + ", " + str(source.y0) + ", " + str(source.z0) + ")")
        t = target.x0
        s = target.z0 - source.z0 - 0.5 * (source.height + target.height)
        a1 = source.radius
        a2 = target.radius
        h1 = source.height
        h2 = target.height
        res, err = dblquad(cmp_yintHoyle, -np.pi/2, np.pi/2, mfun, Mfun, args = (t, s, a1, a2, h1, h2))
        Fx = 2 * G * source.density * target.density * a1 * a2 * res
        return Fx

    def cmpAnaIntFxFirstBad(self, pos1, pos2, _dir = '1->2'):
        #func = lambda r, rp, theta, thetap: self.fx_int(r, rp, theta, thetap, sizes = sizes)
        if self.obj1.y0 != self.obj2.y0:
            raise NotImplementedError('Must rotate reference frame...')

        if np.shape(pos1) != np.shape(pos2):
            raise TypeError("Conflicting positions! Must be lists of 3-elements arrays, with the same shape")
        nr = np.shape(pos1)[0]
        if np.size(np.shape(pos1)) > 1 and nr > 1:
            raise NotImplementedError()
        self.obj1.setOrigin(pos1)
        self.obj2.setOrigin(pos2)
        if _dir == '1->2':
            source = self.obj1
            target = self.obj2
            pos_t = pos2
        elif _dir == '2->1':
            source = self.obj2
            target = self.obj1
            pos_t = pos1
        else:
            raise ValueError("Bad _dir!")
        Rout = source.radius
        Rin = source.innerRadius
        L = source.height
        Rout_p = target.radius
        Rin_p = target.innerRadius
        L_p = target.height
        d_xy = target.x0 - source.x0
        #sizes = {'Rout': Rout, 'L': L, 'Rout_p': Rout_p, 'L_p': L_p, 'd_xy': d_xy}
        
        res, err, outdic = nquad(self.fx_int, [[Rin / Rout, 1], [Rin_p / Rout_p, 1], [0, 2*np.pi], [0, 2*np.pi]], args = (Rout, L, Rout_p, L_p, d_xy))
        Fx = -G * self.obj1.density * self.obj2.density * res
        

    def fx_int(self, r, rp, theta, thetap, Rout, L, Rout_p, L_p, d_xy):
        """
        JB, first attempt at analytic expression: incorrect and too complicated
        """
        rap = Rout_p / Rout
        fac1 = r**2 + rap**2 * rp**2 - 2 * rap * r * rp * np.cos(thetap - theta)
        fac2 = d_xy / Rout * (d_xy / Rout + 2 * rap * rp * np.cos(thetap) - 2 * r * np.cos(theta))
        Rhatsq = Rout**2 * (fac1 + fac2)

        A1sq = (L + L_p + 2 * d_xy)**2
        A2sq = (L - L_p - 2 * d_xy)**2
        A3sq = (L - L_p - 2 * d_xy)**2
        A4sq = (L + L_p - 2 * d_xy)**2

        xfac = rap * rp * np.cos(thetap) - r * np.cos(theta) + d_xy / Rout

        fx = Rout**2 * Rout_p**2 * r * rp * xfac * Rout / (2 * Rhatsq) * (np.sqrt(4 * Rhatsq + A1sq) - np.sqrt(4 * Rhatsq + A2sq) - np.sqrt(4 * Rhatsq + A3sq) + np.sqrt(4 * Rhatsq + A4sq))
        return fx
        


# plotting functions
def plt_accelerationField(axp, ayp, azp, in_it, x, y, z, nx, ny, nz, target, source, Fx, Fy, Fz):
    ax_xy = np.flipud(np.transpose(axp[:,:,int(nz/2)]))
    ay_xy = np.flipud(np.transpose(ayp[:,:,int(nz/2)]))
    az_xy = np.flipud(np.transpose(azp[:,:,int(nz/2)]))
    inax_xy = np.flipud(np.transpose(axp[:,:,int(nz/2)] * in_it[:,:,int(nz/2)]))
    inay_xy = np.flipud(np.transpose(ayp[:,:,int(nz/2)] * in_it[:,:,int(nz/2)]))
    inaz_xy = np.flipud(np.transpose(azp[:,:,int(nz/2)] * in_it[:,:,int(nz/2)]))

    ax_xz = np.flipud(np.transpose(axp[:,int(ny/2),:]))
    ay_xz = np.flipud(np.transpose(ayp[:,int(ny/2),:]))
    az_xz = np.flipud(np.transpose(azp[:,int(ny/2),:]))
    inax_xz = np.flipud(np.transpose(axp[:,int(ny/2),:] * in_it[:,int(ny/2),:]))
    inay_xz = np.flipud(np.transpose(ayp[:,int(ny/2),:] * in_it[:,int(ny/2),:]))
    inaz_xz = np.flipud(np.transpose(azp[:,int(ny/2),:] * in_it[:,int(ny/2),:]))

    ax_x = ax_xy[int(ny/2),:]
    ay_x = ay_xy[int(ny/2),:]
    az_x = az_xy[int(ny/2),:]
    ax_y = ax_xy[:,int(nx/2)]
    ay_y = ay_xy[:,int(nx/2)]
    az_y = az_xy[:,int(nx/2)]
    ax_z = ax_xz[:,int(nx/2)]
    ay_z = ay_xz[:,int(nx/2)]
    az_z = az_xz[:,int(nx/2)]
        
    #plot acceleration in (x,y) plane
    #entire field
    _ax = plt.subplot(351)
    im = _ax.imshow(ax_xy, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ax')
    plots.pltDiskProjection([target.x0, target.y0], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-', axis = _ax)
    plots.pltDiskProjection([source.x0, source.y0], source.radius, hollow = (source.innerRadius > 0), innerRadius = source.innerRadius, color = 'green', linestyle = '-', axis = _ax)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ay = plt.subplot(356, sharex = _ax, sharey = _ax)
    im = _ay.imshow(ay_xy, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ay')
    plots.pltDiskProjection([target.x0, target.y0], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-', axis = _ay)
    plots.pltDiskProjection([source.x0, source.y0], source.radius, hollow = (source.innerRadius > 0), innerRadius = source.innerRadius, color = 'green', linestyle = '-', axis = _ay)
    _ay.set_xlabel('x')
    _ay.set_ylabel('y')
    _az = plt.subplot(3,5,11, sharex = _ax, sharey = _ax)
    im = _az.imshow(az_xy, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('az')
    plots.pltDiskProjection([target.x0, target.y0], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-', axis = _az)
    plots.pltDiskProjection([source.x0, source.y0], source.radius, hollow = (source.innerRadius > 0), innerRadius = source.innerRadius, color = 'green', linestyle = '-', axis = _az)
    _az.set_xlabel('x')
    _az.set_ylabel('y')

    #only in target
    _iax = plt.subplot(352, sharex = _ax, sharey = _ax)
    im = _iax.imshow(inax_xy, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ax')
    plots.pltDiskProjection([target.x0, target.y0], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-', axis = _iax)
    plots.pltDiskProjection([source.x0, source.y0], source.radius, hollow = (source.innerRadius > 0), innerRadius = source.innerRadius, color = 'green', linestyle = '-', axis = _iax)
    _iax.set_xlabel('x')
    _iax.set_ylabel('y')
    _iay = plt.subplot(357, sharex = _iax, sharey = _iax)
    im = _iay.imshow(inay_xy, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ay')
    plots.pltDiskProjection([target.x0, target.y0], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-', axis = _iay)
    plots.pltDiskProjection([source.x0, source.y0], source.radius, hollow = (source.innerRadius > 0), innerRadius = source.innerRadius, color = 'green', linestyle = '-', axis = _iay)
    _iay.set_xlabel('x')
    _iay.set_ylabel('y')
    _iaz = plt.subplot(3,5,12, sharex = _iax, sharey = _iax)
    im = _iaz.imshow(inaz_xy, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('az')
    plots.pltDiskProjection([target.x0, target.y0], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-', axis = _iaz)
    plots.pltDiskProjection([source.x0, source.y0], source.radius, hollow = (source.innerRadius > 0), innerRadius = source.innerRadius, color = 'green', linestyle = '-', axis = _iaz)
    _iaz.set_xlabel('x')
    _iaz.set_ylabel('y')

    #plot acceleration in (x,z) plane
    #entire field
    _ax = plt.subplot(353, sharex = _ax, sharey = _ax)
    im = _ax.imshow(ax_xz, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ax')
    plots.pltRectangleProjection([target.x0, target.z0], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-', axis = _ax)
    plots.pltRectangleProjection([source.x0, target.z0], 2*source.radius, source.height, hollow = (source.innerRadius > 0), innerWidth = 2*source.innerRadius, color = 'green', linestyle = '-', axis = _ax)
    _ax.set_xlabel('x')
    _ax.set_ylabel('z')
    _ay = plt.subplot(358, sharex = _ax, sharey = _ax)
    im = _ay.imshow(ay_xz, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ay')
    plots.pltRectangleProjection([target.x0, target.z0], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-', axis = _ay)
    plots.pltRectangleProjection([source.x0, target.z0], 2*source.radius, source.height, hollow = (source.innerRadius > 0), innerWidth = 2*source.innerRadius, color = 'green', linestyle = '-', axis = _ay)
    _ay.set_xlabel('x')
    _ay.set_ylabel('z')
    _az = plt.subplot(3,5,13, sharex = _ax, sharey = _ax)
    im = _az.imshow(az_xz, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('az')
    plots.pltRectangleProjection([target.x0, target.z0], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-', axis = _az)
    plots.pltRectangleProjection([source.x0, target.z0], 2*source.radius, source.height, hollow = (source.innerRadius > 0), innerWidth = 2*source.innerRadius, color = 'green', linestyle = '-', axis = _az)
    _az.set_xlabel('x')
    _az.set_ylabel('z')

    #only in target
    _iax = plt.subplot(354, sharex = _ax, sharey = _ax)
    im = _iax.imshow(inax_xz, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ax')
    plots.pltRectangleProjection([target.x0, target.z0], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-', axis = _iax)
    plots.pltRectangleProjection([source.x0, target.z0], 2*source.radius, source.height, hollow = (source.innerRadius > 0), innerWidth = 2*source.innerRadius, color = 'green', linestyle = '-', axis = _iax)
    _iax.set_xlabel('x')
    _iax.set_ylabel('z')
    _iay = plt.subplot(359, sharex = _iax, sharey = _iax)
    im = _iay.imshow(inay_xz, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('ay')
    plots.pltRectangleProjection([target.x0, target.z0], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-', axis = _iay)
    plots.pltRectangleProjection([source.x0, target.z0], 2*source.radius, source.height, hollow = (source.innerRadius > 0), innerWidth = 2*source.innerRadius, color = 'green', linestyle = '-', axis = _iay)
    _iay.set_xlabel('x')
    _iay.set_ylabel('z')
    _iaz = plt.subplot(3,5,14, sharex = _iax, sharey = _iax)
    im = _iaz.imshow(inaz_xz, interpolation = 'nearest', extent = (np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('az')
    plots.pltRectangleProjection([target.x0, target.z0], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-', axis = _iaz)
    plots.pltRectangleProjection([source.x0, target.z0], 2*source.radius, source.height, hollow = (source.innerRadius > 0), innerWidth = 2*source.innerRadius, color = 'green', linestyle = '-', axis = _iaz)
    _iaz.set_xlabel('x')
    _iaz.set_ylabel('z')

    #accelerations along x,y,z
    _ax = plt.subplot(355)
    _ax.plot(x, ax_x, label = 'ax')
    _ax.plot(x, ay_x, label = 'ay')
    _ax.plot(x, az_x, label = 'az')
    ym, yM = _ax.get_ylim()
    _ax.plot([target.x0 - target.radius, target.x0 - target.radius], [ym, yM], color = 'blue')
    _ax.plot([target.x0 + target.radius, target.x0 + target.radius], [ym, yM], color = 'blue')
    if target.innerRadius > 0:
        _ax.plot([target.x0 - target.innerRadius, target.x0 - target.innerRadius], [ym, yM], color = 'blue')
        _ax.plot([target.x0 + target.innerRadius, target.x0 + target.innerRadius], [ym, yM], color = 'blue')
    _ax.plot([source.x0 - source.radius, source.x0 - source.radius], [ym, yM], color = 'green')
    _ax.plot([source.x0 + source.radius, source.x0 + source.radius], [ym, yM], color = 'green')
    if source.innerRadius > 0:
        _ax.plot([source.x0 - source.innerRadius, source.x0 - source.innerRadius], [ym, yM], color = 'green')
        _ax.plot([source.x0 + source.innerRadius, source.x0 + source.innerRadius], [ym, yM], color = 'green')
    _ax.legend()
    _ax.set_xlabel('x')

    _ax = plt.subplot(3,5,10)
    _ax.plot(y, ax_y, label = 'ax')
    _ax.plot(y, ay_y, label = 'ay')
    _ax.plot(y, az_y, label = 'az')
    ym, yM = _ax.get_ylim()
    _ax.plot([target.y0 - target.radius, target.y0 - target.radius], [ym, yM], color = 'blue')
    _ax.plot([target.y0 + target.radius, target.y0 + target.radius], [ym, yM], color = 'blue')
    if target.innerRadius > 0:
        _ax.plot([target.y0 - target.innerRadius, target.y0 - target.innerRadius], [ym, yM], color = 'blue')
        _ax.plot([target.y0 + target.innerRadius, target.y0 + target.innerRadius], [ym, yM], color = 'blue')
    _ax.plot([source.y0 - source.radius, source.y0 - source.radius], [ym, yM], color = 'green')
    _ax.plot([source.y0 + source.radius, source.y0 + source.radius], [ym, yM], color = 'green')
    if source.innerRadius > 0:
        _ax.plot([source.y0 - source.innerRadius, source.y0 - source.innerRadius], [ym, yM], color = 'green')
        _ax.plot([source.y0 + source.innerRadius, source.y0 + source.innerRadius], [ym, yM], color = 'green')
    _ax.legend()
    _ax.set_xlabel('y')

    _ax = plt.subplot(3,5,15)
    _ax.plot(z, ax_z, label = 'ax')
    _ax.plot(z, ay_z, label = 'ay')
    _ax.plot(z, az_z, label = 'az')
    ym, yM = _ax.get_ylim()
    _ax.plot([target.z0 - target.height/2, target.z0 - target.height/2], [ym, yM], color = 'blue')
    _ax.plot([target.z0 + target.height/2, target.z0 + target.height/2], [ym, yM], color = 'blue')
    _ax.plot([source.z0 - source.height/2, source.z0 - source.height/2], [ym, yM], color = 'green')
    _ax.plot([source.z0 + source.height/2, source.z0 + source.height/2], [ym, yM], color = 'green')
    _ax.legend()
    _ax.set_xlabel('x')
        
    #info about force
    sFx = "%.2e" % Fx
    sFy = "%.2e" % Fy
    sFz = "%.2e" % Fz
    plt.suptitle('Force Fx, Fy, Fz = ' + sFx + ', ' + sFy + ', ' + sFz)
    plt.show()
    
def plt_targetGrid(x, y, z, nx, ny, nz, min_xoffset, max_xoffset, min_yoffset, max_yoffset, min_zoffset, max_zoffset, target, source, target_gridType):
    if target.kind == 'cylinder' and target_gridType == 'polar':
        #NB: in this case, points are all inside the target
        r = np.copy(x)
        theta = np.copy(y)
        th3, r3, z3 = np.meshgrid(theta, r, z)
        x3 = r3 * np.cos(th3)
        y3 = r3 * np.sin(th3)
        x3 += target.x0
        y3 += target.y0
    else:
        y3, x3, z3 = np.meshgrid(y, x, z) #invert x and y to avoid transpose
        th4p = np.linspace(0, 2 * np.pi, 500)
        ix4p = target.innerRadius * np.cos(th4p) + target.x0
        iy4p = target.innerRadius * np.sin(th4p) + target.y0
        ox4p = target.radius * np.cos(th4p) + target.x0
        oy4p = target.radius * np.sin(th4p) + target.y0
            
    x3 = np.reshape(x3, nx * ny * nz)
    y3 = np.reshape(y3, nx * ny * nz)
    z3 = np.reshape(z3, nx * ny * nz)
    x3m = x3 + min_xoffset
    x3M = x3 + max_xoffset
    y3m = y3 + min_yoffset
    y3M = y3 + max_yoffset
    z3m = z3 + min_zoffset
    z3M = z3 + max_zoffset
    x3vec = [x3m, x3m, x3M, x3M]
    y3vec = [y3m, y3M, y3m, y3M]
    #(x,y) plane
    _ax = plt.subplot(121)
    _ax.plot(x3, y3, marker = '.', linestyle = '', color = 'black')
    plots.pltDiskProjection([target.x0 + min_xoffset, target.y0 + min_yoffset], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-', axis = _ax)
    plots.pltDiskProjection([target.x0 + max_xoffset, target.y0 + min_yoffset], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '--', axis = _ax)
    plots.pltDiskProjection([target.x0 + min_xoffset, target.y0 + max_yoffset], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = '-.', axis = _ax)
    plots.pltDiskProjection([target.x0 + max_xoffset, target.y0 + max_yoffset], target.radius, hollow = (target.innerRadius > 0), innerRadius = target.innerRadius, color = 'blue', linestyle = ':', axis = _ax)
    plots.pltDiskProjection([source.x0, source.y0], source.radius, hollow = (source.innerRadius > 0), innerRadius = source.innerRadius, color = 'green', linestyle = '-', axis = _ax)
    _ax.set_xlabel('x')
    _ax.set_xlabel('y')

    #(x,z) plane
    _az = plt.subplot(122)
    _az.plot(x3, z3, marker = '.', linestyle = '', color = 'black')
    plots.pltRectangleProjection([target.x0 + min_xoffset, target.z0 + min_zoffset], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-', axis = _az)
    plots.pltRectangleProjection([target.x0 + max_xoffset, target.z0 + min_zoffset], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '--', axis = _az)
    plots.pltRectangleProjection([target.x0 + min_xoffset, target.z0 + max_zoffset], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = '-.', axis = _az)
    plots.pltRectangleProjection([target.x0 + max_xoffset, target.z0 + max_zoffset], 2*target.radius, target.height, hollow = (target.innerRadius > 0), innerWidth = 2*target.innerRadius, color = 'blue', linestyle = ':', axis = _az)
    plots.pltRectangleProjection([source.x0, source.z0], 2*source.radius, source.height, hollow = (source.innerRadius > 0), innerWidth = 2*source.innerRadius, color = 'green', linestyle = '-', axis = _az)
    _az.set_xlabel('x')
    _az.set_xlabel('z')
    plt.suptitle('Target grid')
    plt.show()


    
### functions needed for JPU analytic form
def cmpJx(Delta, lmbda):
    #Delta = self.obj1.radius / self.obj2.radius
    #lmbda = self.obj2.height / self.obj2.radius
    #if lmbda <= 1:
    #    raise ValueError("Need lmbda > 1")
    #raise ValueError("Need L > R2")
    Jx, err = quad(Jx_intgd, 0, 1, args = (Delta, lmbda))
    return Jx

def cmpIx(Delta, lmbda):
    #Delta = self.obj1.radius / self.obj2.radius
    #lmbda = self.obj2.height / self.obj2.radius
    #if lmbda <= 1:
    #    raise ValueError("Need lmbda > 1")
    #raise ValueError("Need L > R2")
    Ix, err = quad(Ix_intgd, 0, 1, args = (Delta, lmbda))
    return Ix
    
def Ix_intgd(y, Delta, lmbda):
    f1 = f_e(y, Delta, lmbda) * ellipe(4 * Delta / ((1 + Delta)**2 + lmbda**2 * y**2))
    f2 = f_k(y, Delta, lmbda) * ellipk(4 * Delta / ((1 + Delta)**2 + lmbda**2 * y**2))
    return (f1 + f2) / (8*np.pi)

def Jx_intgd(y, Delta, lmbda):
    f1 = g_e(y, Delta, lmbda) * ellipe(4 * Delta / ((1 + Delta)**2 + lmbda**2 * y**2))
    f2 = g_k(y, Delta, lmbda) * ellipk(4 * Delta / ((1 + Delta)**2 + lmbda**2 * y**2))
    return (f1 + f2) / (8*np.pi * Delta)

def f_e(y, Delta, lmbda):
    #num = 5 - 3 * Delta**4 + 2 * lmbda**2 * y**2 - 3 * lmbda**4 * y**4 - 2 * Delta**2 * (1 + 3 * lmbda**2 * y**2)
    num = 3 - 5 * Delta**4 - 2 * lmbda**2 * y**2 - 5 * lmbda**4 * y**4 + 2 * Delta**2 * (1 - 5 * lmbda**2 * y**2)
    denom = ((1 - Delta)**2 + lmbda**2 * y**2)**2 * ((1 + Delta)**2 + lmbda**2 * y**2)**(3./2)
    return num / denom

def f_k(y, Delta, lmbda):
    #num = -1 + Delta**2 + lmbda**2 * y**2
    num = -1 -2 * Delta**3 + Delta**4 + 2 * Delta * (1 - lmbda**2 * y**2) + 2 * Delta**2 * lmbda**2 * y**2 + lmbda**4 * y**4
    denom = ((1 - Delta)**2 + lmbda**2 * y**2)**2 * ((1 + Delta)**2 + lmbda**2 * y**2)**(3./2)
    return num / denom

def g_e(y, Delta, lmbda):
    #num = 7 * Delta**4 + 6 * Delta**2 * (lmbda**2 * y**2 - 1) - (lmbda**2 * y**2 - 1)**2
    num = 7 * Delta**4 + 6 * Delta**2 * (lmbda**2 * y**2 - 1) - (lmbda**2 * y**2 + 1)**2 #JPU's nb
    denom = ((1 - Delta)**2 + lmbda**2 * y**2)**2 * ((1 + Delta)**2 + lmbda**2 * y**2)**(3./2)
    return num / denom

def g_k(y, Delta, lmbda):
    num = 2 * Delta**3 - Delta**4 - 2 * Delta * (1 + lmbda**2 * y**2) + (1 + lmbda**2 * y**2)**2
    denom = ((1 - Delta)**2 + lmbda**2 * y**2)**2 * ((1 + Delta)**2 + lmbda**2 * y**2)**(3./2)
    return num / denom


#functions needed for analytic form, and also numeric (but not brute force) Yukawa
def cmpIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = 1, yukawa = False, lmbda = None):
    if not yukawa:
        return anaIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
    else:
        return numIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, yukawa = True, lmbda = lmbda)

def cmpIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = 1, yukawa = False, lmbda = None):
    if not yukawa:
        return anaIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, x0 = x0, version = 2)
    else:
        return numIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, yukawa = True, lmbda = lmbda)
    
def cmp_yint_x(theta1, theta2, a1, a2, t, z1min, z1max, z2min, z2max, ia1 = 0, ia2 = 0, yukawa = False, lmbda = None):
    y1 = a1 * np.sin(theta1)
    y2 = a2 * np.sin(theta2)
    Ixz = cmpIxz(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
    #return a1 * a2 * Ixz * np.cos(theta1) * np.cos(theta2)
    #contributions to the total force for hollow cylinders (considered as inner --empty-- cylinders surrounded by outer --full-- cylinders). The "total" force between 2 full cylinders is given by (after integrating over the corresponding y1 and y2) the sum iiIxz + ieIxz + ieIxz + eeIxz, where eeIxz is Ixz between the outer parts of both cylinders (i.e., the "force" between the hollow cylinders), and the other contributors are defined hereafter
    iiIxz = 0 #"force" between inner cylinders
    ieIxz = 0 #"force" between the inner part of the source cylinder and the full target
    eiIxz = 0 #"force" between the full source cylinder and the inner part of the target
    if ia1 > 0 and ia2 > 0:
        #force between inner cylinders
        iy1 = ia1 * np.sin(theta1)
        iy2 = ia2 * np.sin(theta2)
        iiIxz = cmpIxz(iy1, iy2, ia1, ia2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
        ieIxz = cmpIxz(iy1, y2, ia1, a2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
        eiIxz = cmpIxz(y1, iy2, a1, ia2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
    elif ia1 > 0 and ia2 <= 0:
        raise NotImplementedError()
    elif ia1 <= 0 and ia2 > 0:
        raise NotImplementedError()
    else:
        pass
    fac = a1 * a2 * Ixz + ia1 * ia2 * iiIxz - ia1 * a2 * ieIxz - a1 * ia2 * eiIxz
    return fac * np.cos(theta1) * np.cos(theta2)

def cmp_yint_z(theta1, theta2, a1, a2, t, z1min, z1max, z2min, z2max, ia1 = 0, ia2 = 0, yukawa = False, lmbda = None):
    y1 = a1 * np.sin(theta1)
    y2 = a2 * np.sin(theta2)
    Izx = cmpIzx(y1, y2, a1, a2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
    #contributions to the total force for hollow cylinders (considered as inner --empty-- cylinders surrounded by outer --full-- cylinders). The "total" force between 2 full cylinders is given by (after integrating over the corresponding y1 and y2) the sum iiIxz + ieIxz + ieIxz + eeIxz, where eeIxz is Ixz between the outer parts of both cylinders (i.e., the "force" between the hollow cylinders), and the other contributors are defined hereafter
    iiIzx = 0 #"force" between inner cylinders
    ieIzx = 0 #"force" between the inner part of the source cylinder and the full target
    eiIzx = 0 #"force" between the full source cylinder and the inner part of the target
    if ia1 > 0 and ia2 > 0:
        #force between inner cylinders
        iy1 = ia1 * np.sin(theta1)
        iy2 = ia2 * np.sin(theta2)
        iiIzx = cmpIzx(iy1, iy2, ia1, ia2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
        ieIzx = cmpIzx(iy1, y2, ia1, a2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
        eiIzx = cmpIzx(y1, iy2, a1, ia2, t, z1min, z1max, z2min, z2max, yukawa = yukawa, lmbda = lmbda)
    elif ia1 > 0 and ia2 <= 0:
        raise NotImplementedError()
    elif ia1 <= 0 and ia2 > 0:
        raise NotImplementedError()
    else:
        pass
    fac = a1 * a2 * Izx + ia1 * ia2 * iiIzx - ia1 * a2 * ieIzx - a1 * ia2 * eiIzx
    return fac * np.cos(theta1) * np.cos(theta2)

    

#function needed for Hoyle's analytic expression
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
            if ti[i] < 0:
                print('ti<0')
            rij = np.sqrt(ti[i]**2 + y**2 + sj[j]**2)
            mone = (-1)**(i+j)
            f1 = ti[i] * rij
            f2  = np.sign(ti[i]) * (y**2 - sj[j]**2) * np.log(np.abs(ti[i]) + rij)
            if sj[j] != 0:
                f3 = -2 * ti[i] * sj[j] * np.log(np.sign(sj[j]) + rij / np.abs(sj[j]))
            else:
                f3 = 0
            f4 = 2 * y * sj[j] * np.arctan2(ti[i] * sj[j], rij * y)
            inc = mone * (f1 + f2 + f3 + f4)
            Ixz += inc
    Ixz *= -0.5
    return Ixz

# def cmpIxzHoyle_v0(t, y1, y2, s, a1, a2, h1, h2):
#     """
#     Eot-Wash, Hoyle et al (1999)'s Eq. B5
#     """
#     #s = np.abs(s) #not sure about this abs(sj)... if not, then nan!
#     if s < 0:
#         raise NotImplementedError('Pute de s de merde qui doit etre >0!!!')
#     y = y1 - y2
#     t0 = t - np.sqrt(a2**2 - y2**2) - np.sqrt(a1**2 - y1**2)
#     t1 = t - np.sqrt(a2**2 - y2**2) + np.sqrt(a1**2 - y1**2)
#     t2 = t + np.sqrt(a2**2 - y2**2) + np.sqrt(a1**2 - y1**2)
#     t3 = t + np.sqrt(a2**2 - y2**2) - np.sqrt(a1**2 - y1**2)
#     s0 = s
#     s1 = s + h1
#     s2 = s + h1 + h2
#     s3 = s + h2
#     ti = [t0, t1, t2, t3]
#     sj = [s0, s1, s2, s3]
#     Ixz = 0
#     for i in range(4):
#         for j in range(4):
#             rij = np.sqrt(ti[i]**2 + y**2 + sj[j]**2)
#             mone = (-1)**(i+j)
#             f1 = ti[i] * rij
#             if y != 0 and sj[j] != 0:
#                 f2  = np.sign(ti[i]) * (y**2 - sj[j]**2) * np.log((np.abs(ti[i]) + rij) / np.sqrt(y**2 + sj[j]**2))
#             else:
#                 f2 = 0
#             if sj[j] != 0:
#                 f3 = -2 * ti[i] * sj[j] * np.log(1 + rij / np.abs(sj[j])) #not sure about this abs(sj)... if not, then nan!
#             else:
#                 f3 = 0
#             f4 = 2 * y * sj[j] * np.arctan2(ti[i] * sj[j], rij * y)
#             inc = mone * (f1 + f2 + f3 + f4)
#             Ixz += inc
#     Ixz *= -0.5
#     return Ixz
    
def cmp_yintHoyle(theta1, theta2, t, s, a1, a2, h1, h2):
    y1 = a1 * np.sin(theta1)
    y2 = a2 * np.sin(theta2)
    Ixz = cmpIxzHoyle(t, y1, y2, s, a1, a2, h1, h2)
    return Ixz * np.cos(theta1) * np.cos(theta2)


    

def tstQx():
    Delta = 0.1
    Delta2 = 2
    lmbda = np.linspace(1.1, 100, 100)
    nl = np.size(lmbda)
    y = np.linspace(0, 1, 300)
    #plt.plot(y, Jx_intgd(y, Delta, 1))
    #plt.plot(y, Jx_intgd(y, Delta, 13))
    #plt.show()
    Qx_lmbda = np.zeros(nl)
    Qx_lmbda2 = np.zeros(nl)
    for i in range(nl):
        Qx_lmbda[i] = cmpIx(Delta, lmbda[i]) - Delta * cmpJx(Delta, lmbda[i])
        Qx_lmbda2[i] = cmpIx(Delta2, lmbda[i]) - Delta2 * cmpJx(Delta2, lmbda[i])
    plt.loglog(lmbda, np.abs(Qx_lmbda))
    plt.loglog(lmbda, np.abs(Qx_lmbda2))
    plt.xlabel('lmbda')
    plt.ylabel('Qx')
    plt.show()

    Delta = np.linspace(0, 0.65, 100)
    lmbda = np.linspace(0.5, 2, 130)
    nd = np.size(Delta)
    nl = np.size(lmbda)
    Qx_lmbda = np.zeros((nl, nd))
    for i in range(nl):
        for j in range(nd):
            Qx_lmbda[i,j] = cmpIx(Delta[j], lmbda[i]) - Delta[j] * cmpJx(Delta[j], lmbda[i])
    Qx_lmbda = np.flipud(Qx_lmbda)
    #norm = LogNorm(vmin = np.min(Qx_lmbda), vmax = np.max(Qx_lmbda))
    im = plt.imshow(np.abs(Qx_lmbda), cmap = 'jet', aspect = 'auto', extent = [min(Delta), max(Delta), min(lmbda), max(lmbda)])#, norm = norm)
    plt.xlabel('Delta')
    plt.ylabel('lmbda')
    plt.colorbar(im)
    plt.show()

def getMinMax(val, pmin, pmax):
    if pmin is None:
        return val, val
    else:
        return min(val, pmin), max(val, pmax)


def checkBallPairGeometry(ball_1, ball_2, verbose = False):
    okay = True
    r = np.sqrt((ball_2.x0 - ball_1.x0)**2 + (ball_2.y0 - ball_1.y0)**2 + (ball_2.z0 - ball_1.z0)**2)
    if r >= ball_1.radius + ball_2.radius:
        #balls are away from each other. Good in any case
        if verbose:
            print("Balls are away from each other")
    elif ball_1.innerRadius > r + ball_2.radius:
        #ball #1 is hollow and ball_2 is entirely inside it
        if verbose:
            print("ball #1 is hollow and ball_2 is entirely inside it")
    elif ball_2.innerRadius > r + ball_1.radius:
        #ball #2 is hollow and ball_1 is entirely inside it
        if verbose:
            print("ball #2 is hollow and ball_1 is entirely inside it")
    else:
        #all other cases are forbidden (balls overlap)
        okay = False
        if verbose:
            print("Geometry not allowed!")
    return okay


def checkCylinderPairGeometry(cyl1, cyl2, verbose = False):
    r = np.sqrt((cyl2.x0 - cyl1.x0)**2 + (cyl2.y0 - cyl1.y0)**2)
    dh = cyl2.z0 - cyl1.z0
    okay = True
    if cyl1.originType != cyl2.originType:
        raise NotImplementedError()
    elif cyl1.originType == 'centered' and cyl2.originType == 'centered' and np.abs(dh) >= 0.5 * (cyl1.height + cyl2.height):
        #cylinders are above each other. Good in any case
        if verbose:
            print("Cylinders are at different altitudes")
    elif cyl1.originType == 'low' and cyl2.originType == 'low':
        if (cyl1.z0 < cyl2.z0 and np.abs(dh) >= cyl1.height) or (cyl1.z0 > cyl2.z0 and np.abs(dh) >= cyl2.height):
            #cylinders are above each other. Good in any case
            if verbose:
                print("Cylinders are at different altitudes")
    elif r >= cyl1.radius + cyl2.radius:
        #cylinders are away from each other. Good in any case
        if verbose:
            print("cylinders are away from each other")
    elif cyl1.innerRadius > r + cyl2.radius:
        #cylinder #1 is hollow and cyl2 is entirely inside it
        if verbose:
            print("cylinder #1 is hollow and cyl2 is entirely inside it")
    elif cyl2.innerRadius > r + cyl1.radius:
        #cylinder #2 is hollow and cyl1 is entirely inside it
        if verbose:
            print("cylinder #2 is hollow and cyl1 is entirely inside it")
    else:
        #all other cases are forbidden (cylinders overlap)
        okay = False
        if verbose:
            print("Geometry not allowed!")
    if not okay:
        print('-----')
        print(cyl1.x0, cyl1.y0, cyl1.z0, cyl1.height)
        print(cyl2.x0, cyl2.y0, cyl2.z0, cyl2.height)
        print(r, dh, cyl1.height + cyl2.height)
        print(np.abs(dh) >= cyl1.height + cyl2.height)
        print(r >= cyl1.radius + cyl2.radius)
        print(cyl1.innerRadius > r + cyl2.radius)
        print(cyl2.innerRadius > r + cyl1.radius)
    return okay


def checkCylinderBallGeometry(ball, cyl, verbose = False):
    r = np.sqrt((cyl.x0 - ball.x0)**2 + (cyl.y0 - ball.y0)**2)
    dh = cyl.z0 - ball.z0
    okay = True
    if dh >= ball.radius + cyl.height:
        #cylinders and ball are above each other. Good in any case
        if verbose:
            print("Cylinder and ball are at different altitudes")
    elif r >= cyl.radius + ball.radius:
        #cylinders are away from each other. Good in any case
        if verbose:
            print("cylinder and ball are away from each other")
    elif cyl.innerRadius > r + ball.radius:
        #cylinder is hollow and ball is entirely inside it
        if verbose:
            print("cylinder is hollow and ball is entirely inside it")
    elif ball.innerRadius > r + cyl.radius:
        #cylinder #2 is hollow and cyl1 is entirely inside it
        if verbose:
            print("ball is hollow and cylinder is entirely inside it")
    else:
        #all other cases are forbidden (cylinders overlap)
        okay = False
        if verbose:
            print("Geometry not allowed!")
    return okay


##############################################################################
def tstFballs(_dir = '2->1', fast = False, dx_rel = 0.02, dy_rel = 0.04, dz_rel = 0.04, showSource = True, method = 'regular grid'):
    R1 = 1e-2 #m
    iR1 = 0 #m
    rho1 = 2000 #kg/m^3
    R2 = 10e-2 #m
    iR2 = 8e-2 #m
    iR2 = 0 #m
    rho2 = 2000 #kg/m^3
    #iballs = bapa(R1, R2, iR1 = iR1, iR2 = iR2, rho1 = rho1, rho2 = rho2)
    ball_1 = ball.ball(R1, density = rho1, innerRadius = iR1)
    ball_2 = ball.ball(R2, density = rho2, innerRadius = iR2)
    iballs = ballPair(ball_1, ball_2)

    pos1 = []
    pos2 = []
    d = np.linspace(12e-2, 20e-2, 50)
    for di in d:
        pos1.append([di,0,0])
        pos2.append([0,0,0])
    Fx_ana, Fy_ana, Fz_ana = iballs.cmpForce_ana(pos1, pos2, _dir = _dir)
    Fx, Fy, Fz = iballs.cmpForce(pos1, pos2, _dir = _dir, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, method = method, fast = fast)
    
    ax = plt.subplot(311)
    plt.plot(d, Fx, color = 'black')
    plt.plot(d, Fx_ana, color = 'blue', linestyle = '--')
    if showSource:
        ymin, ymax = plt.ylim()
        if _dir == '1->2':
            plt.plot([-R1, -R1], [ymin, ymax], linestyle = '--', color = 'black')
            plt.plot([R1, R1], [ymin, ymax], linestyle = '--', color = 'black')
            if iR1 > 0:
                plt.plot([-iR1, -iR1], [ymin, ymax], linestyle = ':', color = 'black')
                plt.plot([iR1, iR1], [ymin, ymax], linestyle = ':', color = 'black')
        else:
            plt.plot([-R2, -R2], [ymin, ymax], linestyle = '--', color = 'black')
            plt.plot([R2, R2], [ymin, ymax], linestyle = '--', color = 'black')
            if iR2 > 0:
                plt.plot([-iR2, -iR2], [ymin, ymax], linestyle = ':', color = 'black')
                plt.plot([iR2, iR2], [ymin, ymax], linestyle = ':', color = 'black')
    plt.ylabel('Fx [N]')
    plt.subplot(312, sharex = ax)
    plt.plot(d, Fy, color = 'black')
    plt.plot(d, Fy_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fy [N]')
    plt.subplot(313, sharex = ax)
    plt.plot(d, Fz, color = 'black')
    plt.plot(d, Fz_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fz [N]')
    plt.xlabel('d [m]')
    plt.show(block = True)


def tstFcylinders(cset = 1, _axis = 'x', _dir = '2->1', alpha = 1, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', pmax = 15, dx_rel = 0.01, dy_rel = 0.01, dr_rel = 0.01, dtheta_rel = 0.01, dz_rel = 0.01, fast = True, infinite = False, showSource = True, compareWAna = True, plotit = True, source_gridType = 'Cartesian', target_gridType = 'Cartesian', bruteForce = False, plotTGrid = False, plotAField = False, anaMethod = 1, kmax = 50, plotIntgd = False, nk4plot = 500, cmputeStiffness = False, kmax_stiffness = 'auto', cmpTaylor3 = False):
    from micda.stiffness import instrument
    if not _axis in ['x', 'z']:
        raise ValueError('Bad _axis (x or z)')
    xeq = 0
    yeq = 0
    zeq = 0
    if cset == 1:
        #full cylinders side by side
        print('--> Full cylinders side by side')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 0
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 0
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(2.1e-1, 8e-1, 50)
        else:
            xoffset = 2.5e-1
            d = np.linspace(-0.7, 0.7, 50)
        dAna = d
    elif cset == 11:
        #hollow cylinders side by side
        print('--> Hollow cylinders side by side')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 3e-2
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 10e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(2.1e-1, 8e-1, 50)
        else:
            xoffset = 2.5e-1
            d = np.linspace(-0.7, 0.7, 50)
        dAna = d
    elif cset == 121:
        #full cylinders above each other
        print('--> Full cylinders above each other')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 0
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 0
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        #if _axis == 'x':
        #    d = np.linspace(2.1e-1, 8e-1, 50)
        #else:
        if _axis == 'x':
            xoffset = 0
            zoffset = 35e-2
            zeq = zoffset
            #d = np.linspace(-0.7, 0.7, 50)
            d = np.linspace(-1.7, 1.7, 50)
        else:
            xoffset = 0
            zoffset = 0 #35e-2
            zeq = 35e-2
            d = np.linspace(-0.04, 0.04, 50) #+ 35e-2
        dAna = d
    elif cset == 1210:
        #full cylinders above each other
        print('--> Full cylinders above each other')
        stiffnessOK = True
        R1 = 7e-2
        iR1 = 0
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 0
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        #if _axis == 'x':
        #    d = np.linspace(2.1e-1, 8e-1, 50)
        #else:
        if _axis == 'x':
            xoffset = 0
            zoffset = 35e-2
            zeq = zoffset
            #d = np.linspace(-0.7, 0.7, 50)
            d = np.linspace(-1e-3, 1e-3, 50)
        else:
            xoffset = 0
            zoffset = 35e-2
            zeq = zoffset
            d = np.linspace(-0.04, 0.04, 50) #+ 35e-2
        dAna = d
    elif cset == 122:
        #full cylinders above each other
        print('--> Full cylinders above each other')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 0
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 0
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        #if _axis == 'x':
        #    d = np.linspace(2.1e-1, 8e-1, 50)
        #else:
        if _axis == 'x':
            xoffset = 0
            zoffset = -35e-2
            #d = np.linspace(-0.7, 0.7, 50)
            d = np.linspace(-1.7, 1.7, 50)
        else:
            xoffset = 0
            zoffset = 0#35e-2
            zeq = -35e-2
            d = np.linspace(-0.04, 0.04, 50) #- 35e-2
        dAna = d
    elif cset == 1220:
        #full cylinders above each other
        print('--> Full cylinders above each other')
        stiffnessOK = True
        R1 = 7e-2
        iR1 = 0
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 0
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        #if _axis == 'x':
        #    d = np.linspace(2.1e-1, 8e-1, 50)
        #else:
        if _axis == 'x':
            xoffset = 0
            zoffset = -35e-2
            #d = np.linspace(-0.7, 0.7, 50)
            d = np.linspace(-1e-3, 1e-3, 50)
        else:
            xoffset = 0
            zoffset = 0#35e-2
            zeq = -35e-2
            d = np.linspace(-0.04, 0.04, 50) #- 35e-2
        dAna = d
    elif cset == 2:
        print('--> Nested hollow cylinders (#1 in #2)')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 3e-2
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 10e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-2.8e-2, 2.8e-2, 60)
            dAna = np.linspace(-2.8e-2, 2.8e-2, 60)
        else:
            xoffset = 0
            #xoffset = 0.002
            d = np.linspace(-0.7, 0.7, 60)
            dAna = np.linspace(-0.7, 0.7, 60)
    elif cset == 211:
        print('--> Nested hollow cylinders (#1 in #2), #1 longer than #2')
        stiffnessOK = True
        R1 = 7e-2
        iR1 = 3e-2
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 10e-2
        h2 = 5e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-2.8e-4, 2.8e-4, 60)
            dAna = np.linspace(-2.8e-4, 2.8e-4, 60)
        else:
            xoffset = 0
            d = np.linspace(-1e-3, 1e-3, 60)
            dAna = np.linspace(-1e-3, 1e-3, 60)
    elif cset == 212:
        print('--> Nested hollow cylinders (#1 in #2), #1 shorter than #2')
        stiffnessOK = True
        R1 = 7e-2
        iR1 = 3e-2
        h1 = 5e-2
        R2 = 13e-2
        iR2 = 10e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-2.8e-4, 2.8e-4, 60)
            dAna = np.linspace(-2.8e-4, 2.8e-4, 60)
        else:
            xoffset = 0
            d = np.linspace(-1e-3, 1e-3, 60)
            dAna = np.linspace(-1e-3, 1e-3, 60)
    elif cset == 3:
        print('--> Nested hollow cylinders (#2 in #1)')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 6.5e-2
        #iR1 = 6.95e-2
        h1 = 15e-2
        R2 = 2e-2
        iR2 = 1.6e-2
        #iR2 = 1.95e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-2, 1e-2, 10)
            dAna = np.linspace(-1e-2, 1e-2, 60)
        else:
            xoffset = 0
            d = np.linspace(-0.7, 0.7, 10)
            dAna = np.linspace(-0.7, 0.7, 60)
    elif cset == 311:
        print('--> Nested hollow cylinders (#2 in #1), #1 longer than #2')
        stiffnessOK = True
        R1 = 7e-2
        iR1 = 6.5e-2
        #iR1 = 6.95e-2
        h1 = 15e-2
        R2 = 2e-2
        iR2 = 1.6e-2
        #iR2 = 1.95e-2
        h2 = 5e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-4, 1e-4, 60)
            dAna = np.linspace(-1e-4, 1e-4, 60)
        else:
            xoffset = 0
            d = np.linspace(-1e-3, 1e-3, 60)
            dAna = np.linspace(-1e-3, 1e-3, 60)
    elif cset == 312:
        print('--> Nested hollow cylinders (#2 in #1), #1 shorter than #2')
        stiffnessOK = True
        R1 = 7e-2
        iR1 = 6.5e-2
        #iR1 = 6.95e-2
        h1 = 5e-2
        R2 = 2e-2
        iR2 = 1.6e-2
        #iR2 = 1.95e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-4, 1e-4, 10)
            dAna = np.linspace(-1e-4, 1e-4, 60)
        else:
            xoffset = 0
            d = np.linspace(-1e-3, 1e-3, 10)
            dAna = np.linspace(-1e-3, 1e-3, 60)
    elif cset == 4:
        print('--> Full cylinder in hollow cylinder (#1 in #2)')
        R1 = 7e-2
        if anaMethod == 1:
            iR1 = 1e-8 #for method==1
        else:
            iR1 = 0
        h1 = 15e-2
        R2 = 13e-2
        iR2 = 10e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-2.8e-2, 2.8e-2, 10)
            dAna = np.linspace(-2.8e-2, 2.8e-2, 60)
        else:
            xoffset = 0
            #xoffset = 0.002
            d = np.linspace(-0.7, 0.7, 10)
            dAna = np.linspace(-0.7, 0.7, 60)
    elif cset == 5:
        print('--> Full cylinder in hollow cylinder (#2 in #1)')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 6.5e-2
        #iR1 = 6.95e-2
        h1 = 15e-2
        R2 = 2e-2
        if anaMethod == 1:
            iR2 = 1e-8 #for method==1
        else:
            iR2 = 0
        #iR2 = 1.95e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-2, 1e-2, 10)
            dAna = np.linspace(-1e-2, 1e-2, 60)
        else:
            xoffset = 0
            d = np.linspace(-0.7, 0.7, 10)
            dAna = np.linspace(-0.7, 0.7, 60)
    elif cset == 6:
        print('--> Nested thin hollow cylinders (#2 in #1)')
        stiffnessOK = False
        R1 = 7e-2
        iR1 = 6.8e-2
        #iR1 = 6.95e-2
        h1 = 15e-2
        R2 = 2e-2
        iR2 = 1.9e-2
        #iR2 = 1.95e-2
        h2 = 15e-2
        rho1 = 2000
        rho2 = 2000
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-3, 1e-3, 200)
            dAna = np.linspace(-1e-3, 1e-3, 60)
        else:
            xoffset = 0
            d = np.linspace(-0.7, 0.7, 200)
            dAna = np.linspace(-0.7, 0.7, 60)
    elif cset == 'MIC-SUEPIS1-int':
        print('--> Internal electrode cylinder on MIC-SUEP-IS1 TM')
        stiffnessOK = True
        _dir = '2->1'
        R1 = instrument.tm_cyls['IS1-SUEP']['radius_out']
        iR1 = instrument.tm_cyls['IS1-SUEP']['radius_in']
        h1 = instrument.tm_cyls['IS1-SUEP']['height']
        rho1 = instrument.tm_cyls['IS1-SUEP']['density']
        R2 = instrument.is1_silica['inner']['radius_out']
        iR2 = instrument.is1_silica['inner']['radius_in']
        h2 = instrument.is1_silica['inner']['height']
        rho2 = instrument.is1_silica['inner']['density']
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-4, 1e-4, 20)
            dAna = np.linspace(-1e-4, 1e-4, 20)
        else:
            raise NotImplementedError()
    elif cset == 'MIC-SUEPIS1-int-2':
        print('--> Internal electrode cylinder on MIC-SUEP-IS1 TM (smaller cylinder)')
        stiffnessOK = True
        _dir = '2->1'
        R1 = instrument.tm_cyls['IS1-SUEP']['radius_out']
        iR1 = instrument.tm_cyls['IS1-SUEP']['radius_in']
        h1 = instrument.tm_cyls['IS1-SUEP']['height']
        rho1 = instrument.tm_cyls['IS1-SUEP']['density']
        R2 = instrument.is1_silica['inner']['radius_out']
        iR2 = instrument.is1_silica['inner']['radius_in']
        h2 = instrument.is1_silica['inner']['height']
        rho2 = instrument.is1_silica['inner']['density']
        w2 = R2 - iR2
        iR2 /= 1.2
        R2 = iR2 + w2
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-4, 1e-4, 20)
            dAna = np.linspace(-1e-4, 1e-4, 20)
        else:
            raise NotImplementedError()
    elif cset == 'MIC-SUEPIS1-ext':
        print('--> External electrode cylinder on MIC-SUEP-IS1 TM')
        stiffnessOK = True
        _dir = '2->1'
        R1 = instrument.tm_cyls['IS1-SUEP']['radius_out']
        iR1 = instrument.tm_cyls['IS1-SUEP']['radius_in']
        h1 = instrument.tm_cyls['IS1-SUEP']['height']
        rho1 = instrument.tm_cyls['IS1-SUEP']['density']
        R2 = instrument.is1_silica['outer']['radius_out']
        iR2 = instrument.is1_silica['outer']['radius_in']
        h2 = instrument.is1_silica['outer']['height']
        rho2 = instrument.is1_silica['outer']['density']
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-4, 1e-4, 20)
            dAna = np.linspace(-1e-4, 1e-4, 20)
        else:
            raise NotImplementedError()
    elif cset == 'MIC-SUEPIS1-invar_base':
        print('--> Shield invar base on MIC-SUEP-IS1 TM')
        stiffnessOK = True
        _dir = '2->1'
        R1 = instrument.tm_cyls['IS1-SUEP']['radius_out']
        iR1 = instrument.tm_cyls['IS1-SUEP']['radius_in']
        h1 = instrument.tm_cyls['IS1-SUEP']['height']
        rho1 = instrument.tm_cyls['IS1-SUEP']['density']
        R2 = instrument.shield['invar base']['radius_out']
        iR2 = instrument.shield['invar base']['radius_in']
        h2 = instrument.shield['invar base']['height']
        rho2 = instrument.shield['invar base']['density']
        if _axis == 'x':
            zoffset = -instrument.shield['invar base']['z0']
            d = np.linspace(-1e-4, 1e-4, 20)
            dAna = np.linspace(-1e-4, 1e-4, 20)
        else:
            raise NotImplementedError()

    elif cset == 'MIC-SUEPIS1-outer_shield':
        print('--> Outer shield cylinder on MIC-SUEP-IS1 TM')
        stiffnessOK = True
        _dir = '2->1'
        R1 = instrument.tm_cyls['IS1-SUEP']['radius_out']
        iR1 = instrument.tm_cyls['IS1-SUEP']['radius_in']
        h1 = instrument.tm_cyls['IS1-SUEP']['height']
        rho1 = instrument.tm_cyls['IS1-SUEP']['density']
        R2 = instrument.shield['outer']['radius_out']
        iR2 = instrument.shield['outer']['radius_in']
        h2 = instrument.shield['outer']['height']
        rho2 = instrument.shield['outer']['density']
        if _axis == 'x':
            zoffset = 0
            d = np.linspace(-1e-4, 1e-4, 20)
            dAna = np.linspace(-1e-4, 1e-4, 20)
        else:
            raise NotImplementedError()
        
    else:
        raise NotImplementedError()

    if cmputeStiffness and not stiffnessOK:
        raise ValueError("cset " + str(cset) + " not applicable to stiffness estimation")
    if cmpTaylor3 and not stiffnessOK:
        raise ValueError("cset " + str(cset) + " not applicable to Taylor estimation")
    
    pos1 = []
    pos2 = []
    pos1_ana = []
    pos2_ana = []
    if _axis == 'x':
        for di in d:
            if _dir == '2->1':
                pos1.append([di,0,zoffset]) #that's the target
                pos2.append([0,0,0]) #this is the source
            else:
                pos1.append([0,0,0])
                pos2.append([di,0,zoffset])
        for di in dAna:
            if _dir == '2->1':
                #if anaMethod == 1:
                #    pos1_ana.append([di,0,zoffset]) #that's the target
                #    pos2_ana.append([0,0,0]) #this is the source
                #else:
                #offset taken care of with zeq separately
                pos1_ana.append([di,0,0]) #that's the target
                pos2_ana.append([0,0,0]) #this is the source
            else:
                if anaMethod == 1:
                    pos1_ana.append([0,0,0])
                    pos2_ana.append([di,0,zoffset])
                else:
                    pos1_ana.append([0,0,0])
                    pos2_ana.append([di,0,0])
    else:
        for di in d:
            if _dir == '2->1':
                pos1.append([xoffset,0,di]) #that's the target
                pos2.append([0,0,0]) #this is the source
            else:
                pos1.append([0,0,0])
                pos2.append([xoffset,0,di])
        for di in dAna:
            if _dir == '2->1':
                pos1_ana.append([xoffset,0,di]) #that's the target
                pos2_ana.append([0,0,0]) #this is the source
            else:
                pos1_ana.append([0,0,0])
                pos2_ana.append([xoffset,0,di])

    #c = cypa(R1, h1, R2, h2, iR1 = iR1, iR2 = iR2, rho1 = 2000, rho2 = 2000)
    cyl1 = cylinder.cylinder(R1, h1, density = rho1, originType = 'centered', innerRadius = iR1)
    cyl2 = cylinder.cylinder(R2, h2, density = rho2, originType = 'centered', innerRadius = iR2)
    c = cylinderPair(cyl1, cyl2, resolution = {'dx_rel': dx_rel, 'dy_rel': dy_rel, 'dz_rel': dz_rel, 'dr_rel': dr_rel, 'dtheta_rel': dtheta_rel})

    #ky = c.cmpStiffness(_axis, _dir = _dir, lmbda = lmbda, alpha = alpha, kmax = kmax_stiffness, zeq = zeq)
    #print(ky)
    #raise NotImplementedError("Nailed???")
    
    #if cset == 3:
    #    anaFx = c.cmpAnaFx_JPU(d)

    if compareWAna:
        FxAna = np.zeros(np.size(dAna)) #same name if _axis == 'z', cope with it!
        for i in range(np.size(dAna)):
            if _axis == 'x':
                FxAna[i] = c.cmpAnaFx(pos1_ana[i], pos2_ana[i], _dir = _dir, yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot, xeq = xeq, yeq = yeq, zeq = zeq)
            else:
                FxAna[i] = c.cmpAnaFz(pos1_ana[i], pos2_ana[i], _dir = _dir, yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot, xeq = xeq, yeq = yeq, zeq = zeq)
    #if cset == 4 and compareWAna and _axis == 'x':
    #    FxAnaEW = np.zeros(np.size(dAna))
    #    for i in range(np.size(dAna)):
    #        FxAnaEW[i] = c.cmpEWFx(pos1_ana[i], pos2_ana[i], _dir = _dir)
            if cmpTaylor3:
                if _axis == 'x':
                    FxAnaT3 = c.cmpAnaFx3(dAna, _dir = _dir, lmbda = lmbda, alpha = alpha, kmax = kmax, zeq = zeq)
                else:
                    FxAnaT3 = c.cmpAnaFz3(dAna, _dir = _dir, lmbda = lmbda, alpha = alpha, kmax = kmax, zeq = zeq)

    #if not bruteForce:
    Fx, Fy, Fz = c.cmpForce(pos1, pos2, _dir = _dir, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, pmax = pmax, dx_rel = None, dy_rel = None, dr_rel = None, dtheta_rel = None, dz_rel = None, memoryFriendly = False, verbose = False, raisePositionError = False, fast = fast, infiniteCylinder = infinite, source_gridType = source_gridType, target_gridType = target_gridType, plotTGrid = plotTGrid, plotAField = plotAField)#, xeq = xeq, yeq = yeq, zeq = zeq)
    #else:
    #    Fx = np.zeros(np.size(d))
    #    Fy = np.zeros(np.size(d))
    #    Fz = np.zeros(np.size(d))
    #    for i in range(np.size(d)):
    #        Fx[i], Fy[i], Fz[i] = c.cmpForce_BF(pos1[i], pos2[i], _dir = _dir, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, pmax = pmax, dx_rel = dx_rel, dy_rel = dy_rel, dr_rel = dr_rel, dtheta_rel = dtheta_rel, dz_rel = dz_rel, memoryFriendly = False, verbose = False, raisePositionError = False, source_gridType = source_gridType, target_gridType = target_gridType)

    if _axis == 'z':
        Fx = Fz #awkward but easier

    if cmputeStiffness:
        ky = c.cmpStiffness(_axis, _dir = _dir, lmbda = lmbda, alpha = alpha, kmax = kmax_stiffness, zeq = zeq)
        num_ky = (Fx[-1] - Fx[0]) / (d[-1] - d[0])
        if compareWAna:
            ana_ky = (FxAna[-1] - FxAna[0]) / (dAna[-1] - dAna[0])
        else:
            ana_ky = None
        print("---> stiffness")
        print(ky, ana_ky, num_ky)

        
    if plotit:
        if compareWAna:
            ax = plt.subplot(211)
        plt.plot(d, Fx, label = 'num')
        if showSource:
            ymin, ymax = plt.ylim()
            if _axis == 'x':
                if _dir == '1->2':
                    plt.plot([-R1, -R1], [ymin, ymax], linestyle = '--', color = 'black')
                    plt.plot([R1, R1], [ymin, ymax], linestyle = '--', color = 'black')
                    if iR1 > 0:
                        plt.plot([-iR1, -iR1], [ymin, ymax], linestyle = ':', color = 'black')
                        plt.plot([iR1, iR1], [ymin, ymax], linestyle = ':', color = 'black')
                else:
                    plt.plot([-R2, -R2], [ymin, ymax], linestyle = '--', color = 'black')
                    plt.plot([R2, R2], [ymin, ymax], linestyle = '--', color = 'black')
                    if iR2 > 0:
                        plt.plot([-iR2, -iR2], [ymin, ymax], linestyle = ':', color = 'black')
                        plt.plot([iR2, iR2], [ymin, ymax], linestyle = ':', color = 'black')
            else:
                if _dir == '1->2':
                    plt.plot([-0.5*h1, -0.5*h1], [ymin, ymax], linestyle = '--', color = 'black')
                    plt.plot([0.5*h1, 0.5*h1], [ymin, ymax], linestyle = '--', color = 'black')
                else:
                    plt.plot([-0.5*h2, -0.5*h2], [ymin, ymax], linestyle = '--', color = 'black')
                    plt.plot([0.5*h2, 0.5*h2], [ymin, ymax], linestyle = '--', color = 'black')

                    
        plt.ylabel('num. F' + _axis + ' [N]')
        if compareWAna:
            plt.subplot(212, sharex = ax)
            plt.plot(dAna, FxAna, label = 'exact')
            if cmpTaylor3:
                plt.plot(dAna, FxAnaT3, linestyle = '--', label = 'Taylor')
                plt.legend()
            plt.ylabel('ana. F' + _axis + ' [N]')
        plt.xlabel('d [m]')
        plt.show(block = True)

        if compareWAna:
            plt.plot(d, Fx, label = 'num')
            plt.plot(dAna, FxAna, linestyle = '--', label = 'ana')
            if cmpTaylor3:
                plt.plot(dAna, FxAnaT3, linestyle = ':', label = 'Taylor')
            #if cset == 4:
            #    plt.plot(dAna, FxAnaEW, linestyle = ':', label = 'ana (EW)')
            plt.legend()
            plt.ylabel('F' + _axis + ' [N]')
            plt.xlabel('d [m]')
            title = 'cset:' + str(cset)
            if cmputeStiffness:
                kky = "%0.2e" % ky
                nky = "%0.2e" % num_ky
                aky = "%0.2e" % ana_ky
                title += '; k=' + kky + '(num: ' + nky + ', ana: ' + aky + ')'
            plt.suptitle(title)
            plt.show(block = True)
            
    if compareWAna:
        return d, Fx, dAna, FxAna
    else:
        return d, Fx
        
def tstFcylindersY(cset = 2, _axis = 'x', _dir = '2->1', alpha = 1, lmbdas = [0.01, 0.1, 1, 10, 100], method = 'regular grid', pmax = 15, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, fast = True, infinite = False):
    for lmbda in lmbdas:
        d, Fx = tstFcylinders(cset = cset, _axis = _axis, _dir = _dir, alpha = alpha, lmbda = lmbda, getNewton = False, getYukawa = True, method = method, pmax = pmax, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, fast = fast, infinite = infinite, showSource = False, compareWAna = False, plotit = False)
        plt.plot(d, Fx, label = r'$\lambda$ = ' + str(lmbda) + ' m')
    plt.xlabel('d [m]')
    plt.ylabel('F [N]')
    plt.legend()
    plt.show()
        
    
def tstFHollowCylinderOnBall(inout = 'in', alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', pmax = 15, dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, fast = True, showSource = True):
    R1 = 1e-2 #m
    iR1 = 0 #m
    rho1 = 2000 #kg/m^3
    ball_1 = ball.ball(R1, density = rho1, innerRadius = iR1)
    R2 = 8e-2
    iR2 = 5e-2
    h2 = 15e-2
    rho2 = 2000
    cyl2 = cylinder.cylinder(R2, h2, density = rho2, originType = 'centered', innerRadius = iR2)
    s2 = solids2(ball_1, cyl2)
    if inout == 'in':
        d = np.linspace(-0.95*iR2+R1, 0.95*iR2-R1, 50)
    elif inout == 'out':
        dp = np.linspace(1.05*R2+R1, 3*R2, 50)
        dm = np.linspace(-3*R2, -1.05*R2-R1, 50)
        d = np.append(dm, dp)
    else:
        raise ValueError('Bad inout')

    pos1 = []
    pos2 = []
    for di in d:
        pos1.append([di,0,0])
        pos2.append([0,0,0])
    Fx, Fy, Fz = s2.cmpForce(pos1, pos2, _dir = '2->1', dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, method = method, pmax = pmax, fast = fast)

    ax = plt.subplot(311)
    plt.plot(d, Fx, color = 'black')
    if showSource:
        ymin, ymax = plt.ylim()
        plt.plot([-R2, -R2], [ymin, ymax], linestyle = '--', color = 'black')
        plt.plot([R2, R2], [ymin, ymax], linestyle = '--', color = 'black')
        plt.plot([-iR2, -iR2], [ymin, ymax], linestyle = ':', color = 'black')
        plt.plot([iR2, iR2], [ymin, ymax], linestyle = ':', color = 'black')
    #plt.plot(d, Fx_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fx [N]')
    plt.subplot(312, sharex = ax)
    plt.plot(d, Fy, color = 'black')
    #plt.plot(d, Fy_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fy [N]')
    plt.subplot(313, sharex = ax)
    plt.plot(d, Fz, color = 'black')
    #plt.plot(d, Fz_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fz [N]')
    plt.xlabel('d [m]')
    plt.show(block = True)

    
def tst(cset = 1, fast = False):
    if cset == 1:
        R1 = 7e-2
        iR1 = 3e-2
        h1 = 8e-2
        rho1 = 2000
        R2 = 13e-2
        iR2 = 10e-2
        h2 = 15e-2
        rho2 = 2000
        #d = np.array([0.025])
    elif cset == 2:
        R1 = 7e-2
        iR1 = 0
        h1 = 15e-2
        rho1 = 2000
        R2 = 2e-2
        iR2 = 1e-2
        h2 = 150e-2
        rho2 = 2000
        #d = np.array([0.13])
    elif cset == 3:
        R1 = 7e-2
        iR1 = 6.8e-2
        h1 = 15e-2
        rho1 = 2000
        R2 = 2e-2
        iR2 = 1.9e-2
        h2 = 15e-2
        rho2 = 2000
    else:
        raise NotImplementedError()
    
    obj1 = cylinder.cylinder(R1, h1, density = rho1, originType = 'centered', innerRadius = iR1)
    obj2 = cylinder.cylinder(R2, h2, density = rho2, originType = 'centered', innerRadius = iR2)
    #obj1 = ball.ball(R1, density = rho1, innerRadius = iR1)
    #obj2 = ball.ball(R2, density = rho2, innerRadius = iR2)
    
    s2 = solids2(obj1, obj2)
    #s2.checkGeometry([0,0,0], [0.,0.,0.01])
    #s2.plt_horizontal_aslice(1, nradii = 2.5, nx = 30, ny = 30, z = 0, nr = 100, log = False, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', im_method = 'simps', dx_rel = 0.07, dy_rel = 0.07, dz_rel = 0.07, mc_npoints = 1e6, memoryFriendly = False, verbose = False, fast = True)
    #s2.plt_vertical_aslice(1, nradii = 2.5, nheight = 1.5, nx = 30, nz = 30, y = 0, nr = 100, log = False, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'regular grid', im_method = 'simps', dx_rel = 0.07, dy_rel = 0.07, dz_rel = 0.07, mc_npoints = 1e6, memoryFriendly = False, verbose = False, fast = False)

    pos1 = []
    pos2 = []
    d = np.linspace(12e-2, 20e-2, 50)
    for di in d:
        pos1.append([di,0,0])
        pos2.append([0,0,0])
    Fx, Fy, Fz = s2.cmpForce(pos1, pos2, _dir = '2->1', dx_rel = 0.07, dy_rel = 0.07, dz_rel = 0.07, method = 'regular grid', fast = fast)

    ax = plt.subplot(311)
    plt.plot(d, Fx, color = 'black')
    #plt.plot(d, Fx_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fx [N]')
    plt.subplot(312, sharex = ax)
    plt.plot(d, Fy, color = 'black')
    #plt.plot(d, Fy_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fy [N]')
    plt.subplot(313, sharex = ax)
    plt.plot(d, Fz, color = 'black')
    #plt.plot(d, Fz_ana, color = 'blue', linestyle = '--')
    plt.ylabel('Fz [N]')
    plt.xlabel('d [m]')
    plt.show(block = True)
