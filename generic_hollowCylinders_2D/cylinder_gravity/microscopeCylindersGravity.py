"""
import microscopeCylindersGravity as mcg
mcg.cmpForceOnTM('SUEP', 'IS1', 'Y', cmpOnlyActiveAxis = True, onlyCylinders = False, method = 'regular grid')
mcg.cmpForceOnTM('SUEP', 'IS1', 'Y', cmpOnlyActiveAxis = True, onlyCylinders = False, method = 'regular grid', dx_rel = 0.003, dy_rel = 0.003, dz_rel = 0.05)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d
from gravity.solids import cylinder, solidsPair
from micda.stiffness import instrument

# resolution for numerical forces
res_TM1 = {'is1int': {'dx_rel': 0.004, 'dy_rel': 0.004, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
            'is1ext': {'dx_rel': 0.004, 'dy_rel': 0.004, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
               'is2int': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
        'is2ext': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
               'TM': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
        'innerShield': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'outerShield': {'dx_rel': 0.004, 'dy_rel': 0.004, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'base': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'invarBase': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'upperClamp': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'vacuumSystem': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01}}

res_TM2 = {'is1int': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
            'is1ext': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
               'is2int': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
        'is2ext': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
               'TM': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
        'innerShield': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'outerShield': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'base': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'invarBase': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'upperClamp': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01},
    'vacuumSystem': {'dx_rel': 0.005, 'dy_rel': 0.005, 'dz_rel': 0.05, 'dr_rel': 0.01, 'dtheta_rel': 0.01}}


                          
def cmpForceOnTM(_su, _is, _axis, d = None, cmpOnlyActiveAxis = True, onlyCylinders = False, alpha = 1, lmbda = 1, getNewton = True, getYukawa = False, method = 'ana', fast = True, ignoreEdgeEffects = False, returnData = False, anaMethod = 2, kmax = 5000, plotIntgd = False, nk4plot = 500, noShow = False):
    """
    Compute Newtonian force created by all cylinders (assumed fixed) on one test mass. There are 4 silica cylinders and one fixed TM as sources of the gravity. Could do a nice code implementing the geometry of those russian dolls, but instead, just compute the contribution of each cylinder independently, then adds them up (slower but easier...)
    anaMethod -- 1 for Hoyle (Newton only), 2 for Lockerbie/JPU/JB (Newton or Yukawa, but not both at the same time), 3 for Taylor series
    """
    if getNewton and getYukawa:
        raise ValueError("getNewton and getYukawa are exclusive")
    if getYukawa and method == 'ana' and anaMethod == 1:
        raise NotImplementedError("For analytic Yukawa, please use anaMethod = 2 or 3")

    if method == 'ana' and anaMethod in [2,3] and getNewton:
        # this method is general (Newton and Yukawa), so need to fix alpha and lambda to get in the Newton regime
        alpha = 1
        lmbda = 10000
        
    if not _su in ['SUEP', 'SUREF']:
        raise ValueError('Bad _su')
    if _is == 'IS1':
        other_is = 'IS2'
        resolutionDic = res_TM1
    elif _is == 'IS2':
        other_is = 'IS1'
        resolutionDic = res_TM2
    else:
        raise ValueError('Bad _is')

    #transform MICROSCOPE's axes into cylinder gravity code conventions
    if _axis == 'X':
        _ax = 'z'
    elif _axis in ['Y', 'Z']:
        _ax = 'x'
    else:
        raise ValueError("Bad _axis")
        
    silica_is1_int = cylinder.cylinder(instrument.is1_silica['inner']['radius_out'], instrument.is1_silica['inner']['height'], density = instrument.is1_silica['inner']['density'], innerRadius = instrument.is1_silica['inner']['radius_in'], center = [instrument.is1_silica['inner']['x0'], instrument.is1_silica['inner']['y0'], instrument.is1_silica['inner']['z0']])
    silica_is1_ext = cylinder.cylinder(instrument.is1_silica['outer']['radius_out'], instrument.is1_silica['outer']['height'], density = instrument.is1_silica['outer']['density'], innerRadius = instrument.is1_silica['outer']['radius_in'], center = [instrument.is1_silica['outer']['x0'], instrument.is1_silica['outer']['y0'], instrument.is1_silica['outer']['z0']])
    silica_is2_int = cylinder.cylinder(instrument.is2_silica['inner']['radius_out'], instrument.is2_silica['inner']['height'], density = instrument.is2_silica['inner']['density'], innerRadius = instrument.is2_silica['inner']['radius_in'], center = [instrument.is2_silica['inner']['x0'], instrument.is2_silica['inner']['y0'], instrument.is2_silica['inner']['z0']])
    silica_is2_ext = cylinder.cylinder(instrument.is2_silica['outer']['radius_out'], instrument.is2_silica['outer']['height'], density = instrument.is2_silica['outer']['density'], innerRadius = instrument.is2_silica['outer']['radius_in'], center = [instrument.is2_silica['outer']['x0'], instrument.is2_silica['outer']['y0'], instrument.is2_silica['outer']['z0']])

    fixed_tm_id = other_is + '-' + _su
    fixed_tm = cylinder.cylinder(instrument.tm_cyls[fixed_tm_id]['radius_out'], instrument.tm_cyls[fixed_tm_id]['height'], density = instrument.tm_cyls[fixed_tm_id]['density'], innerRadius = instrument.tm_cyls[fixed_tm_id]['radius_in'], center = [instrument.tm_cyls[fixed_tm_id]['x0'], instrument.tm_cyls[fixed_tm_id]['y0'], instrument.tm_cyls[fixed_tm_id]['z0']])
    if fixed_tm.x0 != 0 or fixed_tm.y0 != 0 or fixed_tm.z0 != 0:
        raise NotImplementedError()

    tm_id = _is + '-' + _su
    tm = cylinder.cylinder(instrument.tm_cyls[tm_id]['radius_out'], instrument.tm_cyls[tm_id]['height'], density = instrument.tm_cyls[tm_id]['density'], innerRadius = instrument.tm_cyls[tm_id]['radius_in'], center = [instrument.tm_cyls[tm_id]['x0'], instrument.tm_cyls[tm_id]['y0'], instrument.tm_cyls[tm_id]['z0']])
    if tm.x0 != 0 or tm.y0 != 0 or tm.z0 != 0:
        raise NotImplementedError()

    shield_in = cylinder.cylinder(instrument.shield['inner']['radius_out'], instrument.shield['inner']['height'], density = instrument.shield['inner']['density'], innerRadius = instrument.shield['inner']['radius_in'], center = [instrument.shield['inner']['x0'], instrument.shield['inner']['y0'], instrument.shield['inner']['z0']])
    shield_out = cylinder.cylinder(instrument.shield['outer']['radius_out'], instrument.shield['outer']['height'], density = instrument.shield['outer']['density'], innerRadius = instrument.shield['outer']['radius_in'], center = [instrument.shield['outer']['x0'], instrument.shield['outer']['y0'], instrument.shield['outer']['z0']])
    shield_silica_base = cylinder.cylinder(instrument.shield['silica base']['radius_out'], instrument.shield['silica base']['height'], density = instrument.shield['silica base']['density'], innerRadius = instrument.shield['silica base']['radius_in'], center = [instrument.shield['silica base']['x0'], instrument.shield['silica base']['y0'], instrument.shield['silica base']['z0']])
    shield_invar_base = cylinder.cylinder(instrument.shield['invar base']['radius_out'], instrument.shield['invar base']['height'], density = instrument.shield['invar base']['density'], innerRadius = instrument.shield['invar base']['radius_in'], center = [instrument.shield['invar base']['x0'], instrument.shield['invar base']['y0'], instrument.shield['invar base']['z0']])
    shield_upper_clamp = cylinder.cylinder(instrument.shield['upper clamp']['radius_out'], instrument.shield['upper clamp']['height'], density = instrument.shield['upper clamp']['density'], innerRadius = instrument.shield['upper clamp']['radius_in'], center = [instrument.shield['upper clamp']['x0'], instrument.shield['upper clamp']['y0'], instrument.shield['upper clamp']['z0']])
    shield_vacuum = cylinder.cylinder(instrument.shield['vacuum system']['radius_out'], instrument.shield['vacuum system']['height'], density = instrument.shield['vacuum system']['density'], innerRadius = instrument.shield['vacuum system']['radius_in'], center = [instrument.shield['vacuum system']['x0'], instrument.shield['vacuum system']['y0'], instrument.shield['vacuum system']['z0']])
    
    
    tm_is1int = solidsPair.cylinderPair(tm, silica_is1_int, resolution = resolutionDic['is1int'])
    tm_is1ext = solidsPair.cylinderPair(tm, silica_is1_ext, resolution = resolutionDic['is1ext'])
    tm_is2int = solidsPair.cylinderPair(tm, silica_is2_int, resolution = resolutionDic['is2int'])
    tm_is2ext = solidsPair.cylinderPair(tm, silica_is2_ext, resolution = resolutionDic['is2ext'])
    tm_othertm = solidsPair.cylinderPair(tm, fixed_tm, resolution = resolutionDic['TM'])

    tm_shieldint = solidsPair.cylinderPair(tm, shield_in, resolution = resolutionDic['innerShield'])
    tm_shieldext = solidsPair.cylinderPair(tm, shield_out, resolution = resolutionDic['outerShield'])
    tm_shieldsilicabase = solidsPair.cylinderPair(tm, shield_silica_base, resolution = resolutionDic['base'])
    tm_shieldinvarbase = solidsPair.cylinderPair(tm, shield_invar_base, resolution = resolutionDic['invarBase'])
    tm_shieldupperclamp = solidsPair.cylinderPair(tm, shield_upper_clamp, resolution = resolutionDic['upperClamp'])
    tm_shieldvacuum = solidsPair.cylinderPair(tm, shield_vacuum, resolution = resolutionDic['vacuumSystem'])

    xeq = 0
    yeq = 0
    zeq = 0
    
    #given those instances, we always want 2->1 forces, so that it's pos1 that changes with d
    if d is None:
        if method == 'ana':
            nd = 10
        else:
            #nd = 200
            nd = 10
        #d = np.linspace(-6e-4, 6e-4, 50)
        #d = np.linspace(-1e-4, 1e-4, nd) #larger view
        d = np.linspace(-1e-5, 1e-5, nd) #more suited to MICROSCOPE
        unit4plot = r'[$\mu$m]'
        fac4plot = 1e6
    else:
        unit4plot = '[m]'
        fac4plot = 1
    #positions are defined wrt to center of each cylinder
    pos1 = []
    pos2 = []
    if _ax == 'x':
        for di in d:
            pos1.append([di,0,0])
            pos2.append([0,0,0])
    else:
        for di in d:
            pos1.append([0,0,di])
            pos2.append([0,0,0])

    if method == 'ana':
        #kinda stupid to init those if anaMethod = 3, but cope with it!
        Fx_is1int = np.zeros(np.size(d))
        Fy_is1int = np.zeros(np.size(d))
        Fz_is1int = np.zeros(np.size(d))
        Fx_is1ext = np.zeros(np.size(d))
        Fy_is1ext = np.zeros(np.size(d))
        Fz_is1ext = np.zeros(np.size(d))
        Fx_is2int = np.zeros(np.size(d))
        Fy_is2int = np.zeros(np.size(d))
        Fz_is2int = np.zeros(np.size(d))
        Fx_is2ext = np.zeros(np.size(d))
        Fy_is2ext = np.zeros(np.size(d))
        Fz_is2ext = np.zeros(np.size(d))
        Fx_tm = np.zeros(np.size(d))
        Fy_tm = np.zeros(np.size(d))
        Fz_tm = np.zeros(np.size(d))

        Fx_shieldint = np.zeros(np.size(d))
        Fy_shieldint = np.zeros(np.size(d))
        Fz_shieldint = np.zeros(np.size(d))
        Fx_shieldext = np.zeros(np.size(d))
        Fy_shieldext = np.zeros(np.size(d))
        Fz_shieldext = np.zeros(np.size(d))
        Fx_shieldsilicabase = np.zeros(np.size(d))
        Fy_shieldsilicabase = np.zeros(np.size(d))
        Fz_shieldsilicabase = np.zeros(np.size(d))
        Fx_shieldinvarbase = np.zeros(np.size(d))
        Fy_shieldinvarbase = np.zeros(np.size(d))
        Fz_shieldinvarbase = np.zeros(np.size(d))
        Fx_shieldupperclamp = np.zeros(np.size(d))
        Fy_shieldupperclamp = np.zeros(np.size(d))
        Fz_shieldupperclamp = np.zeros(np.size(d))
        Fx_shieldvacuum = np.zeros(np.size(d))
        Fy_shieldvacuum = np.zeros(np.size(d))
        Fz_shieldvacuum = np.zeros(np.size(d))

        if anaMethod in [1, 2]:
            for i in range(np.size(d)):
                print('ana #' + str(anaMethod), i, np.size(d))
                if not (cmpOnlyActiveAxis and _ax == 'z'):
                    Fx_is1int[i] = tm_is1int.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fx_is1ext[i] = tm_is1ext.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fx_is2int[i] = tm_is2int.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fx_is2ext[i] = tm_is2ext.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fx_tm[i] = tm_othertm.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fx_shieldint[i] = tm_shieldint.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fx_shieldext[i] = tm_shieldext.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    if not onlyCylinders:
                        Fx_shieldsilicabase[i] = tm_shieldsilicabase.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                        Fx_shieldinvarbase[i] = tm_shieldinvarbase.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                        Fx_shieldupperclamp[i] = tm_shieldupperclamp.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                        Fx_shieldvacuum[i] = tm_shieldvacuum.cmpAnaFx(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                if not (cmpOnlyActiveAxis and _ax == 'x'):
                    Fz_is1int[i] = tm_is1int.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fz_is1ext[i] = tm_is1ext.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fz_is2int[i] = tm_is2int.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fz_is2ext[i] = tm_is2ext.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fz_tm[i] = tm_othertm.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fz_shieldint[i] = tm_shieldint.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    Fz_shieldext[i] = tm_shieldext.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                    if not onlyCylinders:
                        Fz_shieldsilicabase[i] = tm_shieldsilicabase.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                        Fz_shieldinvarbase[i] = tm_shieldinvarbase.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                        Fz_shieldupperclamp[i] = tm_shieldupperclamp.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
                        Fz_shieldvacuum[i] = tm_shieldvacuum.cmpAnaFz(pos1[i], pos2[i], _dir = '2->1', yukawa = (getYukawa and not getNewton), lmbda = lmbda, alpha = alpha, method = anaMethod, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot)#, xeq = xeq, yeq = yeq, zeq = zeq)
            
        elif anaMethod == 3:
            if not (cmpOnlyActiveAxis and _ax == 'z'):
                Fx_is1int = tm_is1int.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fx_is1ext = tm_is1ext.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fx_is2int = tm_is2int.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fx_is2ext = tm_is2ext.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fx_tm = tm_othertm.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fx_shieldint = tm_shieldint.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fx_shieldext = tm_shieldext.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                if not onlyCylinders:
                    Fx_shieldsilicabase = tm_shieldsilicabase.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_silica_base.z0)
                    Fx_shieldinvarbase = tm_shieldinvarbase.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_invar_base.z0)
                    Fx_shieldupperclamp = tm_shieldupperclamp.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_upper_clamp.z0)
                    Fx_shieldvacuum = tm_shieldvacuum.cmpAnaFx3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_vacuum.z0)
            if not (cmpOnlyActiveAxis and _ax == 'x'):
                Fz_is1int = tm_is1int.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fz_is1ext = tm_is1ext.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fz_is2int = tm_is2int.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fz_is2ext = tm_is2ext.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fz_tm = tm_othertm.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fz_shieldint = tm_shieldint.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                Fz_shieldext = tm_shieldext.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
                if not onlyCylinders:
                    Fz_shieldsilicabase = tm_shieldsilicabase.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_silica_base.z0)
                    Fz_shieldinvarbase = tm_shieldinvarbase.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_invar_base.z0)
                    Fz_shieldupperclamp = tm_shieldupperclamp.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_upper_clamp.z0)
                    Fz_shieldvacuum = tm_shieldvacuum.cmpAnaFz3(d, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_vacuum.z0)

        else:
            raise ValueError("Bad anaMethod")

    else:
        #brute force numeric
        source_gridType = 'Cartesian'
        target_gridType = 'Cartesian'
        print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from IS1 internal electrodes cylinder")
        Fx_is1int, Fy_is1int, Fz_is1int = tm_is1int.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
        print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from IS1 external electrodes cylinder")
        Fx_is1ext, Fy_is1ext, Fz_is1ext = tm_is1ext.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
        print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from IS2 internal electrodes cylinder")
        Fx_is2int, Fy_is2int, Fz_is2int = tm_is2int.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
        print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from IS2 external electrodes cylinder")
        Fx_is2ext, Fy_is2ext, Fz_is2ext = tm_is2ext.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
        print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from other TM")
        Fx_tm, Fy_tm, Fz_tm = tm_othertm.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)

        print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from internal cylinder shied")
        Fx_shieldint, Fy_shieldint, Fz_shieldint = tm_shieldint.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
        print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from external cylinder shied")
        Fx_shieldext, Fy_shieldext, Fz_shieldext = tm_shieldext.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
        if not onlyCylinders:
            print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from silica base")
            Fx_shieldsilicabase, Fy_shieldsilicabase, Fz_shieldsilicabase = tm_shieldsilicabase.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
            print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from invar base")
            Fx_shieldinvarbase, Fy_shieldinvarbase, Fz_shieldinvarbase = tm_shieldinvarbase.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
            print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from upper clamp")
            Fx_shieldupperclamp, Fy_shieldupperclamp, Fz_shieldupperclamp = tm_shieldupperclamp.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
            print("\n---> microscopeCylindersGravity.cmpForceOnTM: computing force on TM from vacuum system")
            Fx_shieldvacuum, Fy_shieldvacuum, Fz_shieldvacuum = tm_shieldvacuum.cmpForce(pos1, pos2, _dir = '2->1', alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, memoryFriendly = False, verbose = False, raisePositionError = True, fast = fast, infiniteCylinder = ignoreEdgeEffects, source_gridType = source_gridType, target_gridType = target_gridType)
        else:
            Fx_shieldsilicabase = np.zeros(np.size(d))
            Fy_shieldsilicabase = np.zeros(np.size(d))
            Fz_shieldsilicabase = np.zeros(np.size(d))
            Fx_shieldinvarbase = np.zeros(np.size(d))
            Fy_shieldinvarbase = np.zeros(np.size(d))
            Fz_shieldinvarbase = np.zeros(np.size(d))
            Fx_shieldupperclamp = np.zeros(np.size(d))
            Fy_shieldupperclamp = np.zeros(np.size(d))
            Fz_shieldupperclamp = np.zeros(np.size(d))
            Fx_shieldvacuum = np.zeros(np.size(d))
            Fy_shieldvacuum = np.zeros(np.size(d))
            Fz_shieldvacuum = np.zeros(np.size(d))

    Fx = Fx_is1int + Fx_is1ext + Fx_is2int + Fx_is2ext + Fx_tm + Fx_shieldint + Fx_shieldext + Fx_shieldsilicabase + Fx_shieldinvarbase + Fx_shieldupperclamp + Fx_shieldvacuum
    Fy = Fy_is1int + Fy_is1ext + Fy_is2int + Fy_is2ext + Fy_tm + Fy_shieldint + Fy_shieldext + Fy_shieldsilicabase + Fy_shieldinvarbase + Fy_shieldupperclamp + Fy_shieldvacuum
    Fz = Fz_is1int + Fz_is1ext + Fz_is2int + Fz_is2ext + Fz_tm + Fz_shieldint + Fz_shieldext + Fz_shieldsilicabase + Fz_shieldinvarbase + Fz_shieldupperclamp + Fz_shieldvacuum

    if not onlyCylinders:
        Fc_x = [Fx_is1int, Fx_is1ext, Fx_is2int, Fx_is2ext, Fx_tm, Fx_shieldint, Fx_shieldext, Fx_shieldsilicabase, Fx_shieldinvarbase, Fx_shieldupperclamp, Fx_shieldvacuum, Fx]
        Fc_y = [Fy_is1int, Fy_is1ext, Fy_is2int, Fy_is2ext, Fy_tm, Fy_shieldint, Fy_shieldext, Fy_shieldsilicabase, Fy_shieldinvarbase, Fy_shieldupperclamp, Fy_shieldvacuum, Fy]
        Fc_z = [Fz_is1int, Fz_is1ext, Fz_is2int, Fz_is2ext, Fz_tm, Fz_shieldint, Fz_shieldext, Fz_shieldsilicabase, Fz_shieldinvarbase, Fz_shieldupperclamp, Fz_shieldvacuum, Fz]
        
        labels = ['IS1-int', 'IS1-ext', 'IS2-int', 'IS2-ext', 'TM', 'Inner shield', 'Outer shield', 'Silica baseplate', 'Invar baseplate', 'Upper clamp', 'Vacuum system', 'Total']
        colors = ['blue', 'red', 'green', 'orange', 'cyan', 'purple', 'yellow', 'green', 'orange', 'cyan', 'grey', 'black']
        linestyles = ['-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '-']
    else:
        Fc_x = [Fx_is1int, Fx_is1ext, Fx_is2int, Fx_is2ext, Fx_tm, Fx_shieldint, Fx_shieldext, Fx]
        Fc_y = [Fy_is1int, Fy_is1ext, Fy_is2int, Fy_is2ext, Fy_tm, Fy_shieldint, Fy_shieldext, Fy]
        Fc_z = [Fz_is1int, Fz_is1ext, Fz_is2int, Fz_is2ext, Fz_tm, Fz_shieldint, Fz_shieldext, Fz]
        
        labels = ['IS1-int', 'IS1-ext', 'IS2-int', 'IS2-ext', 'TM', 'Inner shield', 'Outer shield', 'Total']
        colors = ['blue', 'red', 'green', 'orange', 'cyan', 'purple', 'yellow', 'black']
        linestyles = ['-', '-', '-', '-', '-', '-', '-', '-']

    if not noShow:
        ncurves = np.size(labels)
        if not cmpOnlyActiveAxis:
            ax = plt.subplot(311)
            for i in range(ncurves):
                plt.plot(d, Fc_x[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
            plt.legend(loc = 'best', ncol = 2, prop = {'size': 8})
            plt.ylabel('Fx [N]')
            plt.subplot(312, sharex = ax)
            for i in range(ncurves):
                plt.plot(d, Fc_y[i], color = colors[i], linestyle = linestyles[i])
            plt.ylabel('Fy [N]')
            plt.subplot(313, sharex = ax)
            for i in range(ncurves):
                plt.plot(d, Fc_z[i], color = colors[i], linestyle = linestyles[i])
            plt.ylabel('Fz [N]')
            plt.xlabel('d [m]')
            plt.show()
        else:
            if _ax == 'x':
                for i in range(ncurves):
                    plt.plot(fac4plot * d, Fc_x[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
                plt.legend(loc = 'upper right', ncol = 2, prop = {'size': 10})
                #plt.ylabel('Fx [N]')
                if getNewton:
                    plt.ylabel(r'$F_{{\rm N},r}$ [N]', fontsize = 11)
                else:
                    plt.ylabel(r'$F_{{\rm Y},r}$ [N]', fontsize = 11)
                #plt.xlabel('$d$ [m]')
                plt.xlabel('$d$ ' + unit4plot, fontsize = 11)
                plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
                plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                plt.xticks(fontsize = 11)
                plt.yticks(fontsize = 11)
                plt.show()
            else:
                for i in range(ncurves):
                    plt.plot(fac4plot * d, Fc_z[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
                plt.legend(loc = 'upper right', ncol = 2, prop = {'size': 10})
                if getNewton:
                    plt.ylabel(r'$F_{{\rm N},x}$ [N]', fontsize = 11)
                else:
                    plt.ylabel(r'$F_{{\rm Y},x}$ [N]', fontsize = 11)
                #plt.xlabel('$d$ [m]')
                plt.xlabel('$d$ ' + unit4plot, fontsize = 11)
                plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
                plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                plt.xticks(fontsize = 11)
                plt.yticks(fontsize = 11)
                plt.show()

    #compute stiffness (only the total one)
    #if method != 'ana' or (method == 'ana' and anaMethod in [1,2]):
    if _ax == 'x':
        k = (Fx[-1] - Fx[0]) / (d[-1] - d[0])
        F = Fx
    else:
        k = (Fz[-1] - Fz[0]) / (d[-1] - d[0])
        F = Fz
    k *= -1 #to be in the convention F = -k.x
    #print('Total stiffness: ' + str(k) + ' N/m')
    #else:
    k_is1int = tm_is1int.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_is1ext = tm_is1ext.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_is2int = tm_is2int.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_is2ext = tm_is2ext.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_tm = tm_othertm.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_shieldint = tm_shieldint.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_shieldext = tm_shieldext.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    if not onlyCylinders:
        k_shieldsilicabase = tm_shieldsilicabase.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_silica_base.z0)
        k_shieldinvarbase = tm_shieldinvarbase.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_invar_base.z0)
        k_shieldupperclamp = tm_shieldupperclamp.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_upper_clamp.z0)
        k_shieldvacuum = tm_shieldvacuum.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_vacuum.z0)
    else:
        k_shieldsilicabase = 0
        k_shieldinvarbase = 0
        k_shieldupperclamp = 0
        k_shieldvacuum = 0
    k_ana = k_is1int + k_is1ext + k_is2int + k_is2ext + k_tm + k_shieldint + k_shieldext + k_shieldsilicabase + k_shieldinvarbase + k_shieldupperclamp + k_shieldvacuum
    k_ana *= -1 #to be in the convention F = -k.x
    print('Total stiffness: ' + str(k) + ' N/m')
    print('                 ' + str(k_ana) + ' N/m')
    
    
    if not returnData:
        return k, k_ana
    else:
        if _ax == 'x':
            return k_ana, d, F, Fc_x
        else:
            return k_ana, d, F, Fc_z


def pltDiffFromTaylor(_su, _is, _axis, alpha = 1, lmbda = 1, getNewton = True, getYukawa = False, kmax = 5000):
    """
    Compare force computed with full integrals and with Taylor series (3rd order). For stiffness, need to comment higher order terms in anaCGrav2's cmpLongitudinalForceTaylor3 and cmpRadialForceTaylor3
    """
    k_anaF, d, FF, Fc_F = cmpForceOnTM(_su, _is, _axis, getNewton = getNewton, getYukawa = getYukawa, anaMethod = 2, kmax = kmax, noShow = True, returnData = True, onlyCylinders = getYukawa)
    k_anaT, d, FT, Fc_T = cmpForceOnTM(_su, _is, _axis, getNewton = getNewton, getYukawa = getYukawa, anaMethod = 3, kmax = kmax, noShow = True, returnData = True, onlyCylinders = getYukawa)
    #transform MICROSCOPE's axes into cylinder gravity code conventions
    if _axis == 'X':
        _ax = 'z'
    elif _axis in ['Y', 'Z']:
        _ax = 'x'
    else:
        raise ValueError("Bad _axis")
    if getNewton:
        labels = ['IS1-int', 'IS1-ext', 'IS2-int', 'IS2-ext', 'TM', 'Inner shield', 'Outer shield', 'Silica baseplate', 'Invar baseplate', 'Upper clamp', 'Vacuum system', 'Total']
        colors = ['blue', 'red', 'green', 'orange', 'cyan', 'purple', 'yellow', 'green', 'orange', 'cyan', 'grey', 'black']
        linestyles = ['-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '-']
    else:
        labels = ['IS1-int', 'IS1-ext', 'IS2-int', 'IS2-ext', 'TM', 'Inner shield', 'Outer shield', 'Total']
        colors = ['blue', 'red', 'green', 'orange', 'cyan', 'purple', 'yellow', 'black']
        linestyles = ['-', '-', '-', '-', '-', '-', '-', '-']
    unit4plot = r'[$\mu$m]'
    fac4plot = 1e6
    ncurves = np.size(labels)
    if _ax == 'x':
        if getNewton:
            ylabel = r'$F_{{\rm N},r}$ [N]'
        else:
            ylabel = r'$F_{{\rm Y},r}$ [N]'
    else:
        if getNewton:
            ylabel = r'$F_{{\rm N},x}$ [N]'
        else:
            ylabel = r'$F_{{\rm Y},x}$ [N]'

    ax = plt.subplot(221)
    for i in range(ncurves):
        plt.plot(fac4plot * d, Fc_F[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
    plt.legend(loc = 'upper right', ncol = 2, prop = {'size': 10})
    plt.ylabel(ylabel, fontsize = 11)
    plt.xlabel('$d$ ' + unit4plot, fontsize = 11)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
    plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)

    plt.subplot(222, sharex = ax)
    for i in range(ncurves):
        plt.plot(fac4plot * d, Fc_T[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
    #plt.legend(loc = 'upper right', ncol = 2, prop = {'size': 10})
    plt.ylabel(ylabel, fontsize = 11)
    plt.xlabel('$d$ ' + unit4plot, fontsize = 11)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
    plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)

    plt.subplot(223, sharex = ax)
    for i in range(ncurves):
        plt.plot(fac4plot * d, Fc_F[i] - Fc_T[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
    #plt.legend(loc = 'upper right', ncol = 2, prop = {'size': 10})
    plt.ylabel('Diff', fontsize = 11)
    plt.xlabel('$d$ ' + unit4plot, fontsize = 11)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
    plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)

    plt.subplot(224, sharex = ax)
    for i in range(ncurves):
        plt.plot(fac4plot * d, 1 - Fc_T[i] / Fc_F[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
    #plt.legend(loc = 'upper right', ncol = 2, prop = {'size': 10})
    plt.ylabel('Rel. diff.', fontsize = 11)
    plt.xlabel('$d$ ' + unit4plot, fontsize = 11)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
    plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.show()

    for i in range(ncurves):
        if getYukawa and labels[i] in ['Outer shield', 'Total']:
            continue
        plt.plot(fac4plot * d, 1 - Fc_T[i] / Fc_F[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
    plt.legend(loc = 'best', ncol = 2, prop = {'size': 10})
    plt.ylabel(r'$1 - F_{\rm stiffness}/F_{\rm exact}$', fontsize = 11)
    plt.xlabel('$d$ ' + unit4plot, fontsize = 11)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
    plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.show()
    

    
        
def cmpStiffness(_su, _is, _axis, onlyCylinders = False, alpha = 1, lmbda = 1, getNewton = True, getYukawa = False, kmax = 5000, plotIntgd = False, nk4plot = 500, returnAll = False):
    """
    Compute Newtonian force created by all cylinders (assumed fixed) on one test mass. There are 4 silica cylinders and one fixed TM as sources of the gravity. Could do a nice code implementing the geometry of those russian dolls, but instead, just compute the contribution of each cylinder independently, then adds them up (slower but easier...)
    anaMethod -- 1 for Hoyle (Newton only), 2 for Lockerbie/JPU/JB (Newton or Yukawa, but not both at the same time), 3 for Taylor series
    """
    if getNewton and getYukawa:
        raise ValueError("getNewton and getYukawa are exclusive")

    if getNewton:
        alpha = 1
        lmbda = 10000
    
    if not _su in ['SUEP', 'SUREF']:
        raise ValueError('Bad _su')
    if _is == 'IS1':
        other_is = 'IS2'
        resolutionDic = res_TM1
    elif _is == 'IS2':
        other_is = 'IS1'
        resolutionDic = res_TM2
    else:
        raise ValueError('Bad _is')

    #transform MICROSCOPE's axes into cylinder gravity code conventions
    if _axis == 'X':
        _ax = 'z'
    elif _axis in ['Y', 'Z']:
        _ax = 'x'
    else:
        raise ValueError("Bad _axis")
        
    silica_is1_int = cylinder.cylinder(instrument.is1_silica['inner']['radius_out'], instrument.is1_silica['inner']['height'], density = instrument.is1_silica['inner']['density'], innerRadius = instrument.is1_silica['inner']['radius_in'], center = [instrument.is1_silica['inner']['x0'], instrument.is1_silica['inner']['y0'], instrument.is1_silica['inner']['z0']])
    silica_is1_ext = cylinder.cylinder(instrument.is1_silica['outer']['radius_out'], instrument.is1_silica['outer']['height'], density = instrument.is1_silica['outer']['density'], innerRadius = instrument.is1_silica['outer']['radius_in'], center = [instrument.is1_silica['outer']['x0'], instrument.is1_silica['outer']['y0'], instrument.is1_silica['outer']['z0']])
    silica_is2_int = cylinder.cylinder(instrument.is2_silica['inner']['radius_out'], instrument.is2_silica['inner']['height'], density = instrument.is2_silica['inner']['density'], innerRadius = instrument.is2_silica['inner']['radius_in'], center = [instrument.is2_silica['inner']['x0'], instrument.is2_silica['inner']['y0'], instrument.is2_silica['inner']['z0']])
    silica_is2_ext = cylinder.cylinder(instrument.is2_silica['outer']['radius_out'], instrument.is2_silica['outer']['height'], density = instrument.is2_silica['outer']['density'], innerRadius = instrument.is2_silica['outer']['radius_in'], center = [instrument.is2_silica['outer']['x0'], instrument.is2_silica['outer']['y0'], instrument.is2_silica['outer']['z0']])

    fixed_tm_id = other_is + '-' + _su
    fixed_tm = cylinder.cylinder(instrument.tm_cyls[fixed_tm_id]['radius_out'], instrument.tm_cyls[fixed_tm_id]['height'], density = instrument.tm_cyls[fixed_tm_id]['density'], innerRadius = instrument.tm_cyls[fixed_tm_id]['radius_in'], center = [instrument.tm_cyls[fixed_tm_id]['x0'], instrument.tm_cyls[fixed_tm_id]['y0'], instrument.tm_cyls[fixed_tm_id]['z0']])
    if fixed_tm.x0 != 0 or fixed_tm.y0 != 0 or fixed_tm.z0 != 0:
        raise NotImplementedError()

    tm_id = _is + '-' + _su
    tm = cylinder.cylinder(instrument.tm_cyls[tm_id]['radius_out'], instrument.tm_cyls[tm_id]['height'], density = instrument.tm_cyls[tm_id]['density'], innerRadius = instrument.tm_cyls[tm_id]['radius_in'], center = [instrument.tm_cyls[tm_id]['x0'], instrument.tm_cyls[tm_id]['y0'], instrument.tm_cyls[tm_id]['z0']])
    if tm.x0 != 0 or tm.y0 != 0 or tm.z0 != 0:
        raise NotImplementedError()

    shield_in = cylinder.cylinder(instrument.shield['inner']['radius_out'], instrument.shield['inner']['height'], density = instrument.shield['inner']['density'], innerRadius = instrument.shield['inner']['radius_in'], center = [instrument.shield['inner']['x0'], instrument.shield['inner']['y0'], instrument.shield['inner']['z0']])
    shield_out = cylinder.cylinder(instrument.shield['outer']['radius_out'], instrument.shield['outer']['height'], density = instrument.shield['outer']['density'], innerRadius = instrument.shield['outer']['radius_in'], center = [instrument.shield['outer']['x0'], instrument.shield['outer']['y0'], instrument.shield['outer']['z0']])
    shield_silica_base = cylinder.cylinder(instrument.shield['silica base']['radius_out'], instrument.shield['silica base']['height'], density = instrument.shield['silica base']['density'], innerRadius = instrument.shield['silica base']['radius_in'], center = [instrument.shield['silica base']['x0'], instrument.shield['silica base']['y0'], instrument.shield['silica base']['z0']])
    shield_invar_base = cylinder.cylinder(instrument.shield['invar base']['radius_out'], instrument.shield['invar base']['height'], density = instrument.shield['invar base']['density'], innerRadius = instrument.shield['invar base']['radius_in'], center = [instrument.shield['invar base']['x0'], instrument.shield['invar base']['y0'], instrument.shield['invar base']['z0']])
    shield_upper_clamp = cylinder.cylinder(instrument.shield['upper clamp']['radius_out'], instrument.shield['upper clamp']['height'], density = instrument.shield['upper clamp']['density'], innerRadius = instrument.shield['upper clamp']['radius_in'], center = [instrument.shield['upper clamp']['x0'], instrument.shield['upper clamp']['y0'], instrument.shield['upper clamp']['z0']])
    shield_vacuum = cylinder.cylinder(instrument.shield['vacuum system']['radius_out'], instrument.shield['vacuum system']['height'], density = instrument.shield['vacuum system']['density'], innerRadius = instrument.shield['vacuum system']['radius_in'], center = [instrument.shield['vacuum system']['x0'], instrument.shield['vacuum system']['y0'], instrument.shield['vacuum system']['z0']])
    
    
    tm_is1int = solidsPair.cylinderPair(tm, silica_is1_int, resolution = resolutionDic['is1int'])
    tm_is1ext = solidsPair.cylinderPair(tm, silica_is1_ext, resolution = resolutionDic['is1ext'])
    tm_is2int = solidsPair.cylinderPair(tm, silica_is2_int, resolution = resolutionDic['is2int'])
    tm_is2ext = solidsPair.cylinderPair(tm, silica_is2_ext, resolution = resolutionDic['is2ext'])
    tm_othertm = solidsPair.cylinderPair(tm, fixed_tm, resolution = resolutionDic['TM'])

    tm_shieldint = solidsPair.cylinderPair(tm, shield_in, resolution = resolutionDic['innerShield'])
    tm_shieldext = solidsPair.cylinderPair(tm, shield_out, resolution = resolutionDic['outerShield'])
    tm_shieldsilicabase = solidsPair.cylinderPair(tm, shield_silica_base, resolution = resolutionDic['base'])
    tm_shieldinvarbase = solidsPair.cylinderPair(tm, shield_invar_base, resolution = resolutionDic['invarBase'])
    tm_shieldupperclamp = solidsPair.cylinderPair(tm, shield_upper_clamp, resolution = resolutionDic['upperClamp'])
    tm_shieldvacuum = solidsPair.cylinderPair(tm, shield_vacuum, resolution = resolutionDic['vacuumSystem'])

    xeq = 0
    yeq = 0
    zeq = 0

    k_is1int = tm_is1int.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_is1ext = tm_is1ext.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_is2int = tm_is2int.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_is2ext = tm_is2ext.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_tm = tm_othertm.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_shieldint = tm_shieldint.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    k_shieldext = tm_shieldext.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = xeq, yeq = yeq, zeq = zeq)
    if not onlyCylinders:
        k_shieldsilicabase = tm_shieldsilicabase.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_silica_base.z0)
        k_shieldinvarbase = tm_shieldinvarbase.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_invar_base.z0)
        k_shieldupperclamp = tm_shieldupperclamp.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_upper_clamp.z0)
        k_shieldvacuum = tm_shieldvacuum.cmpStiffness(_ax, _dir = '2->1', lmbda = lmbda, alpha = alpha, kmax = kmax, xeq = 0, yeq = 0, zeq = -shield_vacuum.z0)
    else:
        k_shieldsilicabase = 0
        k_shieldinvarbase = 0
        k_shieldupperclamp = 0
        k_shieldvacuum = 0
    k_ana = k_is1int + k_is1ext + k_is2int + k_is2ext + k_tm + k_shieldint + k_shieldext + k_shieldsilicabase + k_shieldinvarbase + k_shieldupperclamp + k_shieldvacuum
    k_ana *= -1 #to be in the convention F = -k.x

    ks = {'IS1-int': -k_is1int, 'IS1-ext': -k_is1ext, 'IS2-int': -k_is2int, 'IS2-ext': -k_is2ext, 'TM': -k_tm, 'Inner shield': -k_shieldint, 'Outer shield': -k_shieldext, 'Silica baseplate': -k_shieldsilicabase, 'Invar baseplate': -k_shieldinvarbase, 'Upper clamp': -k_shieldupperclamp, 'Vacuum system': -k_shieldvacuum, 'Total': k_ana}

    if not returnAll:
        return k_ana
    else:
        return k_ana, ks

    
def plt_klambda(_su, _is, _axis, lambda_min, lambda_max, n_lambda, alpha, onlyCylinders = False, kmax = 5000, plotIntgd = False, nk4plot = 500, logscale = True, ylog = False, showForces = False, showkfromforces = False, hold = False, label_in = None, normalizeByMass = False):
    """
    Plot stiffness for Yukawa as a function of lambda. alpha is used trivially as a renormalization
    """
    if logscale:
        lmbdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)
    else:
        lmbdas = np.linspace(lambda_min, lambda_max, n_lambda)

    ks = np.zeros(n_lambda)
    ksf = np.zeros(n_lambda)
    for i in range(n_lambda):
        print(lmbdas[i], _axis)
        ksf[i], ks[i] = cmpForceOnTM(_su, _is, _axis, onlyCylinders = onlyCylinders, alpha = alpha, lmbda = lmbdas[i], getNewton = False, getYukawa = True, method = 'ana', returnData = False, anaMethod = 3, kmax = kmax, plotIntgd = plotIntgd, nk4plot = nk4plot, noShow = (not showForces))

    ksf /= alpha
    ks /= alpha

    if normalizeByMass:
        mass = instrument.mass[_su + '_' + _is]
        ksf /= mass
        ks /= mass
    
    if label_in is None:
        label = 'stiffness'
    else:
        showkfromforces = False
        label = label_in
    if not ylog:
        plt.plot(lmbdas, ks, label = label)
        if showkfromforces:
            plt.plot(lmbdas, ksf, label = 'slope')
    else:
        plt.plot(lmbdas, np.abs(ks), label = label)
        if showkfromforces:
            plt.plot(lmbdas, np.abs(ksf), label = 'slope')
            plt.legend()
    if not hold:
        plt.xlabel('lambda [m]')
        if not normalizeByMass:
            plt.ylabel('kY/alpha [N/m]')
        else:
            plt.ylabel('kY/alpha/M [N/m]')
        if logscale:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.suptitle(_is + '-' + _su)
        plt.show()

        
def plt_klambdas(_axis, lambda_min, lambda_max, n_lambda, onlyCylinders = False, kmax = 5000, logscale = True, ylog = False, normalizeByMass = False):
    """
    Plot stiffness for Yukawa as a function of lambda, for all ISs. alpha is used trivially as a renormalization
    """
    alpha = 1
    plt_klambda('SUEP', 'IS1', _axis, lambda_min, lambda_max, n_lambda, alpha, onlyCylinders = onlyCylinders, kmax = kmax, logscale = logscale, showForces = False, showkfromforces = False, hold = True, label_in = 'IS1-SUEP', ylog = ylog, normalizeByMass = normalizeByMass)
    plt_klambda('SUEP', 'IS2', _axis, lambda_min, lambda_max, n_lambda, alpha, onlyCylinders = onlyCylinders, kmax = kmax, logscale = logscale, showForces = False, showkfromforces = False, hold = True, label_in = 'IS2-SUEP', ylog = ylog, normalizeByMass = normalizeByMass)
    plt_klambda('SUREF', 'IS1', _axis, lambda_min, lambda_max, n_lambda, alpha, onlyCylinders = onlyCylinders, kmax = kmax, logscale = logscale, showForces = False, showkfromforces = False, hold = True, label_in = 'IS1-SUREF', ylog = ylog, normalizeByMass = normalizeByMass)
    plt_klambda('SUREF', 'IS2', _axis, lambda_min, lambda_max, n_lambda, alpha, onlyCylinders = onlyCylinders, kmax = kmax, logscale = logscale, showForces = False, showkfromforces = False, hold = True, label_in = 'IS2-SUREF', ylog = ylog, normalizeByMass = normalizeByMass)

    plt.xlabel(r'$\lambda$ [m]')
    if not normalizeByMass:
        plt.ylabel(r'$k_Y/\alpha$ [N/m]')
    else:
        plt.ylabel(r'$k_Y/\alpha/M$ [ms$^{-2}$/m]')
    if logscale:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlim(lambda_min, lambda_max)
    plt.legend()
    plt.show()



def plt_kbars(_su, _is, _axis, alpha = 1, lmbda = 0.01):
    kN, allN = cmpStiffness(_su, _is, _axis, onlyCylinders = False, alpha = alpha, lmbda = lmbda, getNewton = True, getYukawa = False, kmax = 5000, plotIntgd = False, nk4plot = 500, returnAll = True)
    kY, allY = cmpStiffness(_su, _is, _axis, onlyCylinders = False, alpha = alpha, lmbda = lmbda, getNewton = False, getYukawa = True, kmax = 5000, plotIntgd = False, nk4plot = 500, returnAll = True)

    #z = np.linspace(2.75, 0, 12)
    z_N = np.linspace(5.5, 0, 12)
    z_Y = z_N - 0.25
    k_N = np.zeros(12)
    k_Y = np.zeros(12)
    labels = ['IS1-int', 'IS1-ext', 'IS2-int', 'IS2-ext', 'TM', 'Inner shield', 'Outer shield', 'Silica baseplate', 'Invar baseplate', 'Upper clamp', 'Vacuum system', 'Total']
    colors = ['blue', 'red', 'green', 'orange', 'cyan', 'purple', 'violet', 'lime', 'darkred', 'tomato', 'peachpuff', 'lightsteelblue']
    for i in range(12):
        k_N[i] = allN[labels[i]]
        k_Y[i] = allY[labels[i]]
    
    #ks = {'is1int': -k_is1int, 'is1ext': -k_is1ext, 'is2_int': -k_is2int, 'is2_ext': -k_is2ext, 'TM': -k_tm, 'shied_int': -k_shieldint, 'shield_ext': -k_shieldext, 'silicabase': -k_shieldsilicabase, 'invarbase': -k_shieldinvarbase, 'upperclamp': -k_shieldupperclamp, 'vacuum': -k_shieldvacuum}

    plt.barh(z_N, k_N * 1e7, height = 0.2, color = colors, tick_label = labels)
    plt.barh(z_Y, k_Y * 1e7, height = 0.2, color = colors, hatch = 'x')
    plt.gca().yaxis.grid()
    #yl = plt.ylim()
    #plt.plot([0,0], [yl[0], yl[1]], color = 'black')
    plt.xlabel(r'$k$ [$\times 10^{-7}$ N/m]', fontsize = 11)
    plt.tight_layout()
    plt.show()
    


    
def polyfitForce(degree, _su, _is, _axis, dmin = -1e-5, dmax = 1e-5, nd = 100, onlyCylinders = False, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'ana', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, fast = True, ignoreEdgeEffects = False, plotit = True):
    d = np.linspace(dmin, dmax, nd)
    k, d, F = cmpForceOnTM(_su, _is, _axis, d = d, cmpOnlyActiveAxis = True, onlyCylinders = onlyCylinders, alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, fast = fast, ignoreEdgeEffects = ignoreEdgeEffects, returnData = True)
    p = np.polyfit(d, D, degree, full = False, cov = False)

    x = np.linspace(dmin, dmax, 5*nd)
    fit = np.polyval(p, x)
    residuals = F - np.polyval(p, d)
    print(p)
    if plotit:
        ax = plt.subplot(211)
        plt.plot(d, F, linestyle = '', marker = 'd')
        plt.plot(x, fit)
        plt.ylabel('F [N]')
        plt.subplot(d, residuals)
        plt.ylabel('Residuals')
        plt.xlabel('d [m]')
        plt.show()
        
    

def cmpSurfaceNewtonianPotential(_is, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = True):
    if _is == 'IS1-EP':
        radius_in = 30.801e-3 #m
        radius_out = 39.39e-3
        height = 43.33e-3 #m
        density = 19972 #kg/m^3
    elif _is == 'IS2-EP':
        radius_in = 60.802e-3 #m
        radius_out = 69.401e-3
        height = 79.831e-3 #m
        density = 4420 #kg/m^3
    elif _is == 'IS1-REF':
        radius_in = 30.801e-3 #m
        radius_out = 39.390e-3
        height = 43.331e-3 #m
        density = 19967 #kg/m^3
    elif _is == 'IS2-REF':
        radius_in = 60.799e-3 #m
        radius_out = 69.397e-3
        height = 79.821e-3 #m
        density = 19980 #kg/m^3
    else:
        raise ValueError("Bad cylinder")
        
    c_in = cylinder.cylinder(radius_in, height, density = density, originType = 'centered')
    c_out = cylinder.cylinder(radius_out, height, density = density, originType = 'centered')

    x = radius_out
    y = 0
    z = 0
    U_in = c_in.cmpPotential_XYZ((x, y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, timeit = False)
    U_out = c_out.cmpPotential_XYZ((x, y, z), alpha = alpha, lmbda = lmbda, getNewton = getNewton, getYukawa = getYukawa, method = method, im_method = im_method, dx_rel = dx_rel, dy_rel = dy_rel, dz_rel = dz_rel, mc_npoints = mc_npoints, memoryFriendly = memoryFriendly, verbose = verbose, timeit = False)
    U = U_out - U_in
    return U


def micCylindersPotential():
    for _is in ['IS1-EP', 'IS2-EP', 'IS1-REF', 'IS2-REF']:
        U = cmpSurfaceNewtonianPotential(_is, alpha = 0, lmbda = 1, getNewton = True, getYukawa = False, method = 'tplquad', im_method = 'simps', dx_rel = 0.01, dy_rel = 0.01, dz_rel = 0.01, mc_npoints = 1e6, memoryFriendly = False, verbose = True)
        print("Potential of " + _is + ": ", U)



def tst_barh():
    ys = np.array([0, 2, 4, 6])
    ys2 = [10, 12, 14, 16]
    ws = [1, -0.4, 1.5, 0.3]
    colors = ['blue', 'red', 'green', 'grey']
    tickl = ['be', 'a', 'nice', 'girl']
    plt.barh(ys, ws, color = colors, tick_label = tickl, hatch = '/')
    plt.barh(ys2, ws, color = colors, tick_label = tickl)
    yl = plt.ylim()
    plt.plot([0,0], [yl[0], yl[1]])
    plt.show()
