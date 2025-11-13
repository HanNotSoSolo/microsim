import numpy as np
import matplotlib.pyplot as plt
import micda.stiffness.microscopeCylindersGravity as mcg
from micda.stiffness import instrument, eStiffness


def mkModel(_is, _su, _axis, alpha, lmbda, t = None, x0 = 5e-6, fexc = 1./300, psi = 0, kE = 'theory', kGW = 5e-2, QGW = 50, b = 0, correct4theory_kE = False, plotit = False):
    """
    x0 -- amplitude of excitation (in position) [m]
    fexc -- excitation frequency [Hz]
    psi -- excitation phase
    kE -- electrostatic stiffness (if 'theory', then the theory-expected one is used)
    kGW -- gold wire stiffness
    QGW -- gold wire quality factor
    """
    if t is None:
        t = np.arange(0, 1750, 0.25) #[s]
    mass = instrument.mass[_su + '_' + _is]
    omega_exc = 2. * np.pi * fexc

    if _is == 'IS1':
        _is_int = 1
    elif _is == 'IS2':
        _is_int = 2
    else:
        raise ValueError('Bad _is!')

    #position
    d = x0 * np.sin(omega_exc * t + psi)

    #excitation acceleration
    a_exc = omega_exc**2 * d
    
    #Newton and Yukawa gravity acceleration
    #_dummy, kN = mcg.cmpForceOnTM(_su, _is, _axis, anaMethod = 3, onlyCylinders = False, kmax = 5000, plotIntgd = False, getNewton = True, getYukawa = False, noShow = True)
    #_dummy, kY = mcg.cmpForceOnTM(_su, _is, _axis, anaMethod = 3, onlyCylinders = False, kmax = 5000, plotIntgd = False, getNewton = False, getYukawa = True, alpha = alpha, lmbda = lmbda, noShow = True)
    kN = mcg.cmpStiffness(_su, _is, _axis, onlyCylinders = False, kmax = 5000, getNewton = True, getYukawa = False)
    kY = mcg.cmpStiffness(_su, _is, _axis, onlyCylinders = False, kmax = 5000, getNewton = False, getYukawa = True, alpha = alpha, lmbda = lmbda)
    a_N = kN / mass * d
    a_Y = kY / mass * d

    #electrostatic (control) acceleration
    if kE == 'theory':
        kE, kerr1, kerr2 = eStiffness.cmpStiffnessDistrib(_is = _is_int, _su = _su, errors = False, nsamples = 1)
        kE *= -1
    a_E = kE / mass * d

    if correct4theory_kE:
        kEc, kerr1, kerr2 = eStiffness.cmpStiffnessDistrib(_is = _is_int, _su = _su, errors = False, nsamples = 1)
        kEc *= -1
        a_Ec = kEc / mass * d
        a_E -= a_Ec

    #gold wire
    phi = 1. / QGW
    a_GW = kGW / mass * x0 * np.sin(omega_exc * t + psi - phi)
    
    #total acceleration
    a = a_exc + a_N + a_Y + a_E + a_GW + b
    #a = a_E + b
    
    #plot everybody if required
    if plotit:
        ax = plt.subplot(811)
        plt.plot(t, d)
        plt.ylabel('Pos [m]')
        plt.subplot(812, sharex = ax)
        plt.plot(t, a_exc)
        plt.ylabel('exc')
        plt.subplot(813, sharex = ax)
        plt.plot(t, a_N)
        plt.ylabel('Newton')
        plt.subplot(814, sharex = ax)
        plt.plot(t, a_Y)
        plt.ylabel('Yukawa')
        plt.subplot(815, sharex = ax)
        plt.plot(t, a_E)
        plt.ylabel('Elec.')
        plt.subplot(816, sharex = ax)
        plt.plot(t, a_GW)
        plt.ylabel('GW')
        plt.subplot(817, sharex = ax)
        plt.plot(t, a)
        plt.ylabel('Total')
        plt.xlabel('t [sec]')
        plt.subplot(818)
        plt.plot(d, a)
        plt.xlabel('Position')
        plt.ylabel('Acc.')
        plt.show()
    
    return a


def mkModelK(_is, _su, _axis, b, k0, kw, Q, t = None, x0 = 5e-6, fexc = 1./300, psi = 0, plotit = False):
    """
    x0 -- amplitude of excitation (in position) [m]
    fexc -- excitation frequency [Hz]
    psi -- excitation phase
    k0 -- sum of electrostatic, gravity and excitation stiffnesses
    kw -- gold wire stiffness
    Q -- gold wire quality factor
    """
    if t is None:
        t = np.arange(0, 1750, 0.25) #[s]
    mass = instrument.mass[_su + '_' + _is]
    omega_exc = 2. * np.pi * fexc

    #position
    d = x0 * np.sin(omega_exc * t + psi)

    #acceleration
    kappa_0 = k0 * x0 / mass
    kappa_w = kw * x0 / mass
    phi = 1./Q
    a0 = kappa_0 + kappa_w * (1 - phi**2 / 2 + phi**2 / 24)
    aw = -kappa_w  * phi * (1 - phi**2 / 6)

    a = b + a0 * np.sin(omega_exc * t + psi) + aw * np.cos(omega_exc * t + psi)

    if plotit:
        ax = plt.subplot(311)
        plt.plot(t, d)
        plt.ylabel('Pos [m]')
        plt.subplot(312, sharex = ax)
        plt.plot(t, a)
        plt.ylabel('Acc [m/s^2]')
        plt.xlabel('t [sec]')
        plt.subplot(313)
        plt.plot(d, a)
        plt.xlabel('Position')
        plt.ylabel('Acc.')
        plt.show()
    
    return a


def mkModelKv(_is, _su, _axis, b, k0, kw, Q, Lv, t = None, x0 = 5e-6, fexc = 1./300, psi = 0, plotit = False):
    """
    x0 -- amplitude of excitation (in position) [m]
    fexc -- excitation frequency [Hz]
    psi -- excitation phase
    k0 -- sum of electrostatic, gravity and excitation stiffnesses
    kw -- gold wire stiffness
    Q -- gold wire quality factor
    Lv -- velocity-dependent coefficient
    """
    if t is None:
        t = np.arange(0, 1750, 0.25) #[s]
    mass = instrument.mass[_su + '_' + _is]
    omega_exc = 2. * np.pi * fexc

    #position
    d = x0 * np.sin(omega_exc * t + psi)

    #acceleration
    kappa_0 = k0 * x0 / mass
    kappa_w = kw * x0 / mass
    kappa_Lv = Lv * x0 / mass * omega_exc
    phi = 1./Q
    a0 = kappa_0 + kappa_w * (1 - phi**2 / 2 + phi**2 / 24)
    aw = -kappa_w  * phi * (1 - phi**2 / 6) + kappa_Lv

    a = b + a0 * np.sin(omega_exc * t + psi) + aw * np.cos(omega_exc * t + psi)

    if plotit:
        ax = plt.subplot(311)
        plt.plot(t, d)
        plt.ylabel('Pos [m]')
        plt.subplot(312, sharex = ax)
        plt.plot(t, a)
        plt.ylabel('Acc [m/s^2]')
        plt.xlabel('t [sec]')
        plt.subplot(313)
        plt.plot(d, a)
        plt.xlabel('Position')
        plt.ylabel('Acc.')
        plt.show()
    
    return a
