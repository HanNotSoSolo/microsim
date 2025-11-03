import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from  micda.daio import rdData
import os.path
from scipy.interpolate import interp1d
from micda.stiffness.dataPath import getPath
from pytsa.general.tsa_utils import denoise as pytsa_denoise
from pytsa.analysis.tsa_stats import moving_average
from micda import ioOramic


#dates of excitation for stiffness measurements (seconds from start of the session)
excitationTimes = {'SUEP_IS1_X': {'tmin': 180, 'tmax': 1970},
                       'SUEP_IS1_Y': {'tmin': 2135, 'tmax': 3900},
                       'SUEP_IS1_Z': {'tmin': 4090, 'tmax': 5860},
                       'SUEP_IS2_X': {'tmin': 11880, 'tmax': 13660},
                       'SUEP_IS2_Y': {'tmin': 13835, 'tmax': 15600},
                       'SUEP_IS2_Z': {'tmin': 15780, 'tmax': 17560},
                       'SUREF_IS1_X': {'tmin': 180, 'tmax': 1970},
                       'SUREF_IS1_Y': {'tmin': 2135, 'tmax': 3900},
                       'SUREF_IS2_X': {'tmin': 11880, 'tmax': 13660},
                       'SUREF_IS1_Z': {'tmin': 4090, 'tmax': 5860},
                       'SUREF_IS2_Y': {'tmin': 13835, 'tmax': 15600},
                       'SUREF_IS2_Z': {'tmin': 15780, 'tmax': 17560}}

def getData(_su, _is, _axis, mode, plotit = False, plotAll = False, raw_times = False, correct4inertia = False):
    if _su == 'SUEP':
        if mode == 'HRM':
            #_path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_538_TSPL_DECSINHR_01_SUEP/N0b_S_01/SUEP/"
            session = 538
        elif mode == 'FRM':
            session = 550
        else:
            raise ValueError('Bad mode')
    elif _su == 'SUREF':
        if mode == 'HRM':
            #_path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_626_TSPL_DECSINHR_01_SUREF/N0b_S_01/SUREF/"
            session = 626
        elif mode == 'FRM':
            session = 638
        else:
            raise ValueError('Bad mode')
    else:
        raise ValueError("Bad SU", _su)
    _path = getPath(session, _su)
    _path = os.path.join(_path, _su)

    if not _is in ['IS1', 'IS2']:
        raise ValueError("Bad IS", _is)

    if not _axis in ['X', 'Y', 'Z']:
        raise ValueError("Bad axis", _axis)

    if _axis in ['Y', 'Z']:
        acc_file = os.path.join(_path, _is, 'Acceleration' + _axis + '.bin')
    else:
        acc_file = os.path.join(_path, _is, 'AccelerationXscaa.bin')
    pos_file = os.path.join(_path, _is, 'Position' + _axis + 'M.bin')
    
    t_acc, acc4Hz, m_acc = rdData(acc_file)
    t_pos, pos, m_pos = rdData(pos_file)
    tpos0 = t_pos[0]
    t_acc -= t_acc[0]
    t_pos -= t_pos[0]

    f = interp1d(t_acc, acc4Hz, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    acc = f(t_pos)

    if plotAll:
        ax = plt.subplot(211)
        plt.plot(t_pos, pos)
        plt.ylabel('Position [m]')
        plt.subplot(212, sharex = ax)
        plt.plot(t_acc, acc4Hz, label = '4Hz')
        plt.plot(t_pos, acc, label = 'Interp 1Hz')
        plt.legend(loc = 'upper right')
        plt.xlabel('t [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.show(block = True)

    tmin = excitationTimes[_su + '_' + _is + '_' + _axis]['tmin']
    tmax = excitationTimes[_su + '_' + _is + '_' + _axis]['tmax']
    gd = np.where((t_pos >= tmin) & (t_pos <= tmax))[0]
    acc = acc[gd]
    pos = pos[gd]
    t = t_pos[gd]
    if not raw_times:
        t -= t[0]
    else:
        t += tpos0

    if correct4inertia:
        clin = inertia(_su, _is, _axis, mode)
        #clin.cmp_inertiaMatrix()
        clin.cmpInertiaAcc()
        if _axis == 'X':
            acc -= clin.a_x
        elif _axis == 'Y':
            acc -= clin.a_y
        elif _axis == 'Z':
            acc -= clin.a_z
        else:
            raise ValueError('WTH?!?')
        
    if plotit:
        ax = plt.subplot(311)
        plt.plot(t, pos)
        plt.ylabel('Position [m]')
        plt.subplot(312, sharex = ax)
        plt.plot(t, acc)
        plt.xlabel('t [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.subplot(313)
        plt.plot(pos, acc)
        plt.xlabel('x [m]')
        plt.ylabel('acc [m/s^2]')
        plt.tight_layout()
        plt.show(block = True)

        ax1 = plt.subplot(111)
        ax1.plot(t, pos * 1e6)
        ax1.set_xlabel('t [s]', fontsize = 13)
        ax1.set_ylabel(r'Position [$\mu$m]', color = 'blue', fontsize = 13)
        ax1.tick_params(axis='y', labelcolor='blue', labelsize = 13)
        ax1.tick_params(axis='x', labelsize = 13)
        ax2 = ax1.twinx()
        ax2.plot(t, acc * 1e7, color = 'red')
        ax2.set_ylabel(r'Acceleration [$\times 10^{-7}$ m/s$^2$]', color = 'red', fontsize = 13)
        ax1.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-3,2), useOffset = False)
        ax2.tick_params(axis='y', labelcolor='red', labelsize = 13)
        plt.tight_layout()
        plt.show()

        #plt.subplot(122)
        plt.plot(pos * 1e6, acc * 1e7)
        plt.xlabel(r'Position [$\mu$m]', fontsize = 13)
        plt.ylabel(r'Acceleration [$\times 10^{-7}$ m/s$^2$]', fontsize = 13)
        plt.tick_params(axis = 'both', labelsize = 13)
        plt.tight_layout()
        plt.show()

        #remove trend in pos-acc relation to see if we can better detect a Lissajous curve (to see phase between pos and acc)
        #p = np.polyfit(pos, acc, 1)
        #pv = np.poly1d(p)
        #res = acc - pv(pos)
        #plt.plot(pos, res, marker = '.', linestyle = '')
        #plt.show()

    return t, acc, pos


def getVpVd(_su, _is, _axis, mode, useConstants = [None, None], plotit = False):
    #if _su == 'SUEP':
    #    _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_538_TSPL_DECSINHR_01_SUEP/N0b_S_01/SUEP/"
    #elif _su == 'SUREF':
    #    _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_626_TSPL_DECSINHR_01_SUREF/N0b_S_01/SUREF/"
    #else:
    #    raise ValueError("Bad SU", _su)

    if _su == 'SUEP':
        if mode == 'HRM':
            #_path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_538_TSPL_DECSINHR_01_SUEP/N0b_S_01/SUEP/"
            session = 538
        elif mode == 'FRM':
            session = 550
        else:
            raise ValueError('Bad mode')
    elif _su == 'SUREF':
        if mode == 'HRM':
            #_path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_626_TSPL_DECSINHR_01_SUREF/N0b_S_01/SUREF/"
            session = 626
        elif mode == 'FRM':
            session = 638
        else:
            raise ValueError('Bad mode')
    else:
        raise ValueError("Bad SU", _su)
    _path = getPath(session, _su)
    _path = os.path.join(_path, _su)
    
    if not _is in ['IS1', 'IS2']:
        raise ValueError("Bad IS", _is)

    if not _axis in ['X', 'Y', 'Z']:
        raise ValueError("Bad axis", _axis)

    Vpin = useConstants[0]
    Vdin = useConstants[1]
    
    tmin = excitationTimes[_su + '_' + _is + '_' + _axis]['tmin']
    tmax = excitationTimes[_su + '_' + _is + '_' + _axis]['tmax']

    Vd_file = os.path.join(_path, _is, 'TensionVD.bin')
    Vp_file = os.path.join(_path, _is, 'TensionVP.bin')
    
    t, Vd, m = rdData(Vd_file)
    t, Vp, m = rdData(Vp_file)
    t -= t[0]

    if plotit:
        ax = plt.subplot(211)
        plt.plot(t, Vd)
        plt.ylabel('Vd [V]')
        plt.subplot(212, sharex = ax)
        plt.plot(t, Vp)
        plt.xlabel('t [s]')
        plt.ylabel('Vp [V]')
        plt.tight_layout()
        plt.show(block = True)
    
    gd = np.where((t >= tmin) & (t <= tmax))[0]
    Vd = Vd[gd]
    Vp = Vp[gd]
    t = t[gd]
    t -= t[0]

    if plotit:
        ax = plt.subplot(211)
        plt.plot(t, Vd)
        plt.ylabel('Vd [V]')
        plt.subplot(212, sharex = ax)
        plt.plot(t, Vp)
        plt.xlabel('t [s]')
        plt.ylabel('Vp [V]')
        plt.tight_layout()
        plt.show(block = True)

    if Vdin is None:
        Vd = np.mean(Vd)
    else:
        if np.abs(np.mean(Vd) - Vdin) / Vdin > 0.1:
            print("WARNING! Measured and input Vd differ by more than 10%", np.mean(Vd), Vdin)
        Vd = np.mean(Vd)
        #Vd = Vdin
    if Vpin is None:
        Vp = np.mean(Vp)
    else:
        if np.abs(np.mean(Vp) - Vpin) / Vpin > 0.1:
            print("WARNING! Measured and input Vp differ by more than 10%", np.mean(Vp), Vpin)
        Vp = np.mean(Vp)
        #Vp = Vpin
        
    return Vp, Vd


def getPositionVoltages(_su, _is, _axis, plotit = False):
    if _su == 'SUEP':
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_538_TSPL_DECSINHR_01_SUEP/N0b_S_01/SUEP/"
    elif _su == 'SUREF':
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_626_TSPL_DECSINHR_01_SUREF/N0b_S_01/SUREF/"
    else:
        raise ValueError("Bad SU", _su)

    if not _is in ['IS1', 'IS2']:
        raise ValueError("Bad IS", _is)

    if not _axis in ['X', 'Y', 'Z']:
        raise ValueError("Bad axis", _axis)

    tmin = excitationTimes[_su + '_' + _is + '_' + _axis]['tmin']
    tmax = excitationTimes[_su + '_' + _is + '_' + _axis]['tmax']

    V1_file = os.path.join(_path, _is, 'Position' + _axis + '1.bin')
    V2_file = os.path.join(_path, _is, 'Position' + _axis + '2.bin')
    
    t, V1, m = rdData(V1_file)
    t, V2, m = rdData(V2_file)
    t -= t[0]

    if plotit:
        ax = plt.subplot(211)
        plt.plot(t, V1)
        plt.ylabel('V1 [V]')
        plt.subplot(212, sharex = ax)
        plt.plot(t, V2)
        plt.xlabel('t [s]')
        plt.ylabel('V2 [V]')
        plt.tight_layout()
        plt.show(block = True)
    
    gd = np.where((t >= tmin) & (t <= tmax))[0]
    V1 = V1[gd]
    V2 = V2[gd]
    t = t[gd]
    t -= t[0]

    if plotit:
        ax = plt.subplot(211)
        plt.plot(t, V1)
        plt.ylabel('V1 [V]')
        plt.subplot(212, sharex = ax)
        plt.plot(t, V2)
        plt.xlabel('t [s]')
        plt.ylabel('V2 [V]')
        plt.tight_layout()
        plt.show(block = True)

    return t, V1, V2


class inertia:
    def __init__(self, _su, _is, _axis, mode, forcePath = None):
        if not forcePath:
            self.t, self.acc, self.pos = getData(_su, _is, _axis, mode, raw_times = True)
            if _su == 'SUEP':
                if mode == 'HRM':
                    #_path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_538_TSPL_DECSINHR_01_SUEP/N0b_S_01/SUEP/"
                    session = 538
                elif mode == 'FRM':
                    session = 550
                else:
                    raise ValueError('Bad mode')
            elif _su == 'SUREF':
                if mode == 'HRM':
                    #_path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_626_TSPL_DECSINHR_01_SUREF/N0b_S_01/SUREF/"
                    session = 626
                elif mode == 'FRM':
                    session = 638
                else:
                    raise ValueError('Bad mode')
            else:
                raise ValueError("Bad SU", _su)
            self._path = getPath(session, _su)
            #self.n0_path = os.path.join(self._path, _su)

        else:
            self.n0_data_path = forcePath
            self._path = self.n0_data_path
            fname = os.path.join(forcePath, _su, _is, 'AccelerationXscaa.bin')
            t_acc, acc, n0mask = rdData(fname)
            fname = os.path.join(forcePath, _su, _is, 'PositionXM.bin')
            t_pos, pos, n0mask = rdData(fname)
            tpos0 = t_pos[0]
            t_acc -= t_acc[0]
            t_pos -= t_pos[0]
            f = interp1d(t_acc, acc, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            acc = f(t_pos)
            t_pos += tpos0
            self.t = t_pos
            self.acc = acc
            self.pos = pos
            

    def cmpInertiaAcc(self, denoise = False, use_oramic = False):
        """
        Since we are in the special case of stiffness measurement sessions and orders of magnitude estimates show that only the [In]O_satO_TM term is not negligible, we use only this one...
        """
        self.cmp_inertiaMatrix(use_oramic = use_oramic) #in satellite's frame
        xp = self.pos[0] #in SU frame but wrt to O_sat
        yp = self.pos[1] + 0.17 #in SU frame but wrt to O_sat
        zp = self.pos[2] #in SU frame but wrt to O_sat
        self.a_x = self.inMatrix_xz * yp - self.inMatrix_yz * zp - self.inMatrix_zz * xp
        self.a_y = self.inMatrix_xx * yp - self.inMatrix_xy * zp - self.inMatrix_xz * xp
        self.a_z = self.inMatrix_xy * yp - self.inMatrix_yy * zp - self.inMatrix_yz * xp
        if denoise:
            t, self.a_x = pytsa_denoise(self.t, self.a_x, prop = {'type': 'butter', 'fs': 4, 'fcut': 0.005, 'n': 30, 'butterN': 2, 'butter_fc': 0.002}, correctMean = False, plotit = False, showSpectra = False)
            t_tmp, self.a_y = pytsa_denoise(self.t, self.a_y, prop = {'type': 'butter', 'fs': 4, 'fcut': 0.005, 'n': 30, 'butterN': 2, 'butter_fc': 0.002}, correctMean = False, plotit = False, showSpectra = False)
            t_tmp, self.a_z = pytsa_denoise(self.t, self.a_z, prop = {'type': 'butter', 'fs': 4, 'fcut': 0.005, 'n': 30, 'butterN': 2, 'butter_fc': 0.002}, correctMean = False, plotit = False, showSpectra = False)


    def pltInAcc(self, xlim = None, denoise = False, use_oramic = False, correct4mean = False):
        self.cmpInertiaAcc(denoise = denoise, use_oramic = use_oramic)
        t = self.t - self.t[0]
        ax = plt.subplot(311)
        if not correct4mean:
            plt.plot(t, self.a_x)
        else:
            plt.plot(t, self.a_x - np.mean(self.a_x))
        plt.ylabel('a_x [ms^-2]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(312, sharex = ax)
        #plt.plot(t, self.a_y)
        if not correct4mean:
            plt.plot(t, self.a_y)
        else:
            plt.plot(t, self.a_y - np.mean(self.a_y))
        plt.ylabel('a_y [ms^-2]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(313, sharex = ax)
        #plt.plot(t, self.a_z)
        if not correct4mean:
            plt.plot(t, self.a_z)
        else:
            plt.plot(t, self.a_z - np.mean(self.a_z))
        plt.ylabel('a_z [ms^-2]')
        plt.xlabel('t [sec]')
        if xlim is not None:
            plt.xlim(xlim)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.tight_layout()
        plt.show()

    def rdOramic(self):
        attpath = os.path.join(self.n0_data_path, 'Mission')
        for f in os.listdir(attpath):
            if 'MIC_CECT_PRECISE_ATTITUDE' in f:
                fname = os.path.join(attpath, f)
                self.gotORAMICattitude = True
                break

        if not self.gotORAMICattitude:
            raise IOError("micda.rdStiffness.rdOramic: could not find precise attitude file (ORAMIC). Will try with coarse APID108 attitude.")
        a = ioOramic.attitude(fname = fname)
        a.read()
        if not np.array_equal(self.t, a.t):
            print("WARNING! micda.core.getOmega' + axis + ': Conflicting self.t and local t! Interpolating.")
            f = interp1d(a.t, a.omegax, bounds_error = False, fill_value = 'extrapolate')
            self.oramic_omegax = f(self.t)
            f = interp1d(a.t, a.omegay, bounds_error = False, fill_value = 'extrapolate')
            self.oramic_omegay = f(self.t)
            f = interp1d(a.t, a.omegaz, bounds_error = False, fill_value = 'extrapolate')
            self.oramic_omegaz = f(self.t)
            f = interp1d(a.t, a.dotomegax, bounds_error = False, fill_value = 'extrapolate')
            self.oramic_dotomegax = f(self.t)
            f = interp1d(a.t, a.dotomegay, bounds_error = False, fill_value = 'extrapolate')
            self.oramic_dotomegay = f(self.t)
            f = interp1d(a.t, a.dotomegaz, bounds_error = False, fill_value = 'extrapolate')
            self.oramic_dotomegaz = f(self.t)
        else:
            self.oramic_omegax = a.omegax
            self.oramic_omegay = a.omegay
            self.oramic_omegaz = a.omegaz
            self.oramic_dotomegax = a.dotomegax
            self.oramic_dotomegay = a.dotomegay
            self.oramic_dotomegaz = a.dotomegaz
        
    def cmp_inertiaMatrix(self, rotateYZ = False, use_oramic = False):
        #angular velocities are about instrument's axes
        t, OmegaX, n0mask = self.getOmegaZ()
        OmegaX *= -1
        t, OmegaY, n0mask = self.getOmegaX()
        t, OmegaZ, n0mask = self.getOmegaY()
        OmegaZ *= -1
        if not use_oramic:
            dot_OmegaX = np.gradient(OmegaX, t[1] - t[0])
            dot_OmegaY = np.gradient(OmegaY, t[1] - t[0])
            dot_OmegaZ = np.gradient(OmegaZ, t[1] - t[0])
        else:
            OmegaX = self.oramic_omegax
            OmegaY = self.oramic_omegay
            OmegaZ = self.oramic_omegaz
            dot_OmegaX = -self.oramic_dotomegax
            dot_OmegaY = self.oramic_dotomegay
            dot_OmegaZ = -self.oramic_dotomegaz

        self.inMatrix_xx = -(OmegaY**2 + OmegaZ**2)
        self.inMatrix_xy = OmegaX * OmegaY - dot_OmegaZ
        self.inMatrix_xz = OmegaX * OmegaZ + dot_OmegaY
        self.inMatrix_yx = OmegaX * OmegaY + dot_OmegaZ
        self.inMatrix_yy = -(OmegaX**2 + OmegaZ**2)
        self.inMatrix_yz = OmegaY * OmegaZ - dot_OmegaX
        self.inMatrix_zx = OmegaX * OmegaZ - dot_OmegaY
        self.inMatrix_zy = OmegaY * OmegaZ + dot_OmegaX
        self.inMatrix_zz = -(OmegaX**2 + OmegaY**2)
        if rotateYZ:
            self.inMatrix_xy *= -1 
            self.inMatrix_xz *= -1 
            self.inMatrix_yx *= -1 
            self.inMatrix_zx *= -1

    def getOmegaI(self, axis, verbose = False):
        """
        Get angular velocity about satellite's I-axis
        Look for MIC_CECT_PRECISE_ATTITUDE file. If not found, look for VitesseAngulaireIEstime.bin from APID108
        """
        if self.t is None:
            raise AttributeError("Would be better to have t before fetching attitude!")
        
        fname = os.path.join(self._path, 'VitesseAngulaire' + axis.upper() + 'Estime.bin')
        if not os.path.isfile(fname):
            raise IOError("No coarse attitude found! Could not find file " + fname)
        t, Omega, n0mask = rdData(fname)
        #t, Omega, n0mask = self.selectTimes(t, Omega, n0mask, 'getOmegaI')
        if not np.array_equal(self.t, t):
            #actually, we do not care because we just need the amplitude of the sine
            if verbose:
                print("WARNING! micda.core.getOmega' + axis + ': Conflicting self.t and local t! Interpolating.")
                print("t", np.size(t))
                print(t)
                print("self.t", np.size(self.t))
                print(self.t)
            f = interp1d(t, Omega, bounds_error = False, fill_value = 'extrapolate')
            Omega = f(self.t)
            t = np.copy(self.t)
        t -= t[0] #to help lsq below, with decent times
        return t, Omega, None
        
    def getOmegaX(self, verbose = False):
        """Get angular velocity about satellite's X-axis (=Yaxis of the intrument), needed to estimate alpha0 for Dy"""
        return self.getOmegaI('X', verbose = verbose)
        
    def getOmegaY(self, verbose = False):
        """Get angular velocity about satellite's Y-axis (=-Zaxis of the intrument), needed to estimate alpha0 for Dy"""
        return self.getOmegaI('Y', verbose = verbose)

    def getOmegaZ(self, verbose = False):
        """Get angular velocity about satellite's Z-axis (=-Xaxis of the intrument), needed to estimate alpha0 for ac12 (thetacz) and ac13 (thetacy)"""
        return self.getOmegaI('Z', verbose = verbose)

    def pltOmegas(self):
        t, OmegaX, _dummy = self.getOmegaX()
        t, OmegaY, _dummy = self.getOmegaY()
        t, OmegaZ, _dummy = self.getOmegaZ()
        t -= t[0]
        ax = plt.subplot(311)
        plt.plot(t, OmegaX)
        plt.ylabel('Omega_X [s^-1]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(312, sharex = ax)
        plt.plot(t, OmegaY)
        plt.ylabel('Omega_Y [s^-1]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(313, sharex = ax)
        plt.plot(t, OmegaZ)
        plt.ylabel('Omega_Z [s^-1]')
        plt.xlabel('t [sec]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.tight_layout()
        plt.show()

    def pltIn(self, use_oramic = False):
        self.cmp_inertiaMatrix(use_oramic = use_oramic)
        t = self.t - self.t[0]
        ax = plt.subplot(331)
        plt.plot(t, self.inMatrix_xx)
        plt.ylabel('Inxx [s^-2]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(332, sharex = ax)
        plt.plot(t, self.inMatrix_xy)
        plt.ylabel('Inxy [s^-2]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(333, sharex = ax)
        plt.plot(t, self.inMatrix_xz)
        plt.ylabel('Inxz [s^-2]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(335, sharex = ax)
        plt.plot(t, self.inMatrix_yy)
        plt.ylabel('Inyy [s^-2]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(336, sharex = ax)
        plt.plot(t, self.inMatrix_yz)
        plt.ylabel('Inyz [s^-2]')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.subplot(339, sharex = ax)
        plt.plot(t, self.inMatrix_zz)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset = False))
        plt.gca().ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.ylabel('Inzz [s^-2]')
        plt.xlabel('t [sec]')
        plt.tight_layout()
        plt.show()
        
