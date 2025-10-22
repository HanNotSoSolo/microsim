import numpy as np
import matplotlib.pyplot as plt
from micda.stiffness import rdStiffnessData, instrument, eStiffness
from micda.stiffness.model import mkModel, mkModelK, mkModelKv
import micda.stiffness.microscopeCylindersGravity as mcg
#from pytsa.analysis.paramsEstimation import fitSingleSine, lsq
import pytsa.analysis.paramsEstimation as pe
from uncertainties import ufloat
from scipy.optimize import minimize, fsolve, brentq
import emcee
import corner
from scipy.fftpack import fftfreq
try:
    from pyfftw.interfaces.scipy_fftpack import fft
except:
    from scipy.fftpack import fft

kQratios = {'X-SUEP': ufloat(1.1e-3, 0.2e-3),
          'Y-SUEP': ufloat(0.3e-3,0.5e-3),
          'Z-SUEP': ufloat(1.3e-3,2e-3),
          'X-SUREF': ufloat(0.07e-3, 0.03e-3),
          'Y-SUREF': ufloat(0.2e-3,0.3e-3),
          'Z-SUREF': ufloat(0.7e-3, 0.5e-3)}

######################################
### MCMC general config and models ###
######################################
#starting values (kE and b set dynamically)
alpha_0 = 1
lmbda_0 = 0.01
#kE_0 = 'theory'
kGW_0 = 0
QGW_0 = 1000
#b_0 = 0
Lv_0 = 0

#allowed ranges (bias and k_E based pre-fit and theory)
alpha_range = [-1e5, 1e5]
lmbda_range = [1e-4, 100]
kGW_range = [-0.1, 0.1]
#kGW_range = [-1, 1]
QGW_range = [1, 1000]
Lv_range = [-0.1, 0.1]
#Lv_range = [-1, 1]

def funY(_is, _su, _axis, t, alpha, lmbda, kE, kGW, QGW, b, psi = 0, correct4theory_kE = False, plotit = False):
    """
    Function to fit Yukawa parameters at the same time as other stiffnesses
    """
    #print("---> fun, ", alpha, lmbda, kE, kGW, QGW, b, psi)
    if correct4theory_kE:
        raise NotImplementedError("Gotta consider mode for theory electrostatic stiffness...")
    if lmbda < 0:
        raise ValueError('lmbda must be > 0')
    Gamma = mkModel(_is, _su, _axis, alpha, lmbda, t = t, x0 = 5e-6, fexc = 1./300, psi = psi, kE = kE, kGW = kGW, QGW = QGW, b = b, correct4theory_kE = correct4theory_kE, plotit = plotit)
    return Gamma

def funK(_is, _su, _axis, t, k0, kw, Q, b, psi, plotit = False):
    """
    Function to fit stiffnesses only (with the same model as that used for least-squares)
    """
    if Q < 0:
        raise ValueError("Q must be > 0")
    Gamma = mkModelK(_is, _su, _axis, b, k0, kw, Q, t = t, x0 = 5e-6, fexc = 1./300, psi = psi, plotit = plotit)
    return Gamma

def funKv(_is, _su, _axis, t, k0, kw, Q, Lv, b, psi, plotit = False):
    """
    Function to fit stiffnesses only (with the same model as that used for least-squares)
    """
    if Q < 0:
        raise ValueError("Q must be > 0")
    Gamma = mkModelKv(_is, _su, _axis, b, k0, kw, Q, Lv, t = t, x0 = 5e-6, fexc = 1./300, psi = psi, plotit = plotit)
    return Gamma

def log_likelihood(modelType, theta, x, y, yerr, _is = None, _su = None, _axis = None, psi = 0, correct4theory_kE = False):
    if modelType == 'stiffness':
        return log_likelihoodK(theta, x, y, yerr, _is = _is, _su = _su, _axis = _axis, psi = psi)
    elif modelType == 'Yukawa':
        return log_likelihoodY(theta, x, y, yerr, _is = _is, _su = _su, _axis = _axis, psi = psi, correct4theory_kE = correct4theory_kE)
    if modelType == 'stiffness+damping':
        return log_likelihoodKv(theta, x, y, yerr, _is = _is, _su = _su, _axis = _axis, psi = psi)
    else:
        raise NotImplementedError()

def log_likelihoodY(theta, x, y, yerr, _is = None, _su = None, _axis = None, psi = 0, correct4theory_kE = False):
    if _is is None or _su is None or _axis is None:
        raise TypeError("_is, _su and _axis must be set")
    alpha, lmbda, kE, kGW, QGW, b = theta
    model = funY(_is, _su, _axis, x, alpha, lmbda, kE, kGW, QGW, b, psi = psi, correct4theory_kE = correct4theory_kE)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

def log_likelihoodK(theta, x, y, yerr, _is = None, _su = None, _axis = None, psi = 0):
    if _is is None or _su is None or _axis is None:
        raise TypeError("_is, _su and _axis must be set")
    k0, kw, Q, b = theta
    model = funK(_is, _su, _axis, x, k0, kw, Q, b, psi)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

def log_likelihoodKv(theta, x, y, yerr, _is = None, _su = None, _axis = None, psi = 0):
    if _is is None or _su is None or _axis is None:
        raise TypeError("_is, _su and _axis must be set")
    k0, kw, Q, Lv, b = theta
    model = funKv(_is, _su, _axis, x, k0, kw, Q, Lv, b, psi)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))



##########################
### Least-square model ###
##########################
def lsqMatrixModel(t, omega, psi, phase, plotit = False):
    """Prepare least-square model (1, sin(omega*t+psi), cos(omega*t+psi)"""
    s = np.sin(omega * t + psi)
    c = np.cos(omega * t + psi)
    b = np.ones(np.size(t))

    if phase == 'acceleration':
        s *= -1
        c *= -1
    
    model = np.vstack((b, np.vstack((s, c))))
    
    if plotit:
        nd = np.size(t)
        ax = plt.subplot(311)
        plt.plot(t, model[0,:])
        plt.ylabel('b')
        plt.subplot(312, sharex = ax)
        plt.plot(t, model[1,:])
        plt.ylabel('sin')
        plt.subplot(313, sharex = ax)
        plt.plot(t, model[2,:])
        plt.ylabel('cos')
        plt.xlabel('t [sec]')
        plt.show()
    
    return model


def _fun4Q(x, a):
    return x**4 - a * x**3 - x**2 / 2 + 1./24

def _trigfun4Q(x, a):
    return x * np.cos(1./x) - a

def solve4Q(a, Qstart = 0.1, Qend = 1000, step = 0.5, trig = False, plotit = False):
    """
    Iterative solver for quartic equation giving wire's Q.
    scipy.optimize.fsolve is too sensitive on guess roots, and analytic solutions seem cumbersome. This iterative technique is particularly easy and may be ok.
    """
    if trig:
        _fun = _trigfun4Q
    else:
        _fun = _fun4Q
    roots = []
    Q0 = Qstart
    Qloc = Q0 + step
    while Qloc <= Qend:
        if _fun(Q0, a) * _fun(Qloc, a) < 0:
            iroot = brentq(_fun, Q0, Qloc, args = (a))
            roots.append(iroot)
            Q0 = Qloc
        Qloc += step
    roots = np.array(roots)
    #print(roots)
    #froots = fsolve(_fun4Q, roots, args = (a))
    #print(froots)
    if plotit:
        x = np.linspace(-13, 123, 5000)
        plt.plot(x, _fun(x, a))
        plt.show()
    return roots

def solve4Q_mc(a, nsamples = 1000, Qstart = 0.1, Qend = 1000, step = 0.5, trig = False):
    """
    Iterative solver for quartic equation giving wire's Q.
    scipy.optimize.fsolve is too sensitive on guess roots, and analytic solutions seem cumbersome. This iterative technique is particularly easy and may be ok.
    """
    root1 = []
    root2 = []
    a0 = np.random.randn(nsamples) * a.std_dev + a.nominal_value
    for i in range(nsamples):
        roots_i = solve4Q(a0[i], Qstart = Qstart, Qend = Qend, step = step, trig = trig)
        nroots = np.size(roots_i)
        if nroots <= 0:
            continue
        root1.append(roots_i[0])
        if nroots == 2:
            root2.append(roots_i[1])
        elif nroots > 2:
            raise NotImplementedError('Arf, more than 2 roots!')
    root1 = np.array(root1)
    root2 = np.array(root2)
    Q1_mean = np.mean(root1)
    Q1_err = np.std(root1, ddof = 1)
    Q2_mean = np.mean(root2)
    Q2_err = np.std(root2, ddof = 1)
    Q1 = ufloat(Q1_mean, Q1_err)
    Q2 = ufloat(Q2_mean, Q2_err)
    return [Q1, Q2]
    


class suk(object):
    def __init__(self, _su, _is, _axis, mode, correct4inertia = True, correct4theory_kE = False, e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 1e-4, plotData = False):
        if not _axis in ['X', 'Y', 'Z']:
            raise ValueError('Bad _axis')
        if not _is in ['IS1', 'IS2']:
            raise ValueError('Bad _is')
        if not _su in ['SUEP', 'SUREF']:
            raise ValueError('Bad _su')
        if not mode in ['HRM', 'FRM']:
            raise ValueError('Bad mode.')
        
        self.correct4theory_kE = correct4theory_kE
        self._su = _su
        self._is = _is
        self._axis = _axis
        self.mode = mode
        if _is == 'IS1':
            _is_int = 1
        elif _is == 'IS2':
            _is_int = 2
        else:
            raise ValueError('Bad _is!')
        self.t, self.acc, self.pos = rdStiffnessData.getData(_su, _is, _axis, mode, plotit = plotData, correct4inertia = correct4inertia)
        self.omega_exc = 2.* np.pi / 300 #period of 300 sec
        self.m = instrument.mass[_su + '_' + _is]
        self.getErrors()
        self.getExcitationCharacteristics()
        self.getAccelerationSine()
        if self._axis == 'X':
            kE_0 = 0
            kerr1 = e_adhoc_absolute_error
            kerr2 = 0
            self.kE_0 = kE_0
            self.kE_rms0 = kerr1
        else:
            kE_0, kerr1, kerr2 = eStiffness.cmpStiffnessDistrib(_is = _is_int, _su = _su, mode = self.mode, errors = True, nsamples = 10000, bias = e_bias, adhoc_relative_error = e_adhoc_relative_error, adhoc_absolute_error = e_adhoc_absolute_error, plotDistrib = False)
            self.kE_0 = -kE_0 #careful to convention!!!
            self.kE_rms0 = kerr1 #could be used to define a range
        #self.kE_bounds = [-(np.abs(self.kE_0) + 5 * self.kE_rms0), np.abs(self.kE_0) + 5 * self.kE_rms0] #don't apply huge constraints for maximum likelihood...
        self.kE_bounds = [-(np.abs(self.kE_0) + 15 * self.kE_rms0), np.abs(self.kE_0) + 15 * self.kE_rms0] #don't apply huge constraints for maximum likelihood...
        #self.kE_priors = [self.kE_0 - 5 * self.kE_rms0, self.kE_0 + 5 * self.kE_rms0] #...but for MCMC priors, want to be more restrictive

    def getTheoryElectrostaticStiffness(self):
        return ufloat(self.kE_0, self.kE_rms0)

    def getNewtonStiffness(self):
        kN = mcg.cmpStiffness(self._su, self._is, self._axis, onlyCylinders = False, kmax = 5000, getNewton = True, getYukawa = False)
        return kN

    def getYukawaStiffness(self, alpha, lmbda):
        kY = mcg.cmpStiffness(self._su, self._is, self._axis, onlyCylinders = False, kmax = 5000, getNewton = False, getYukawa = True, alpha = alpha, lmbda = lmbda)
        return kY
    
    def fitAndCorrectEarthg(self, polynomial = True):
        """
        Not trivial to get Earth gravity acceleration for those sessions (at least from actens), but it should be enough to fit it with a polynomial
        """
        if polynomial:
            raise ValueError("This is pretty bad!!!")
            p = np.polyfit(self.t, self.acc, 2)
            gfit = np.polyval(p, self.t)
            res = self.acc - gfit
            ax = plt.subplot(211)
            plt.plot(self.t, self.acc)
            plt.plot(self.t, gfit)
            plt.ylabel('acc')
            plt.subplot(212, sharex = ax)
            plt.plot(self.t, res)
            plt.show(block = True)
            self.acc = res
        else:
            raise NotImplementedError()

    def mkMock(self, k0, kw, Q):
        print("mkMock: warning! replacing acceleration...")
        self.mockStiffness(k0)
        self.addMockWire(kw, Q)
        
    def mockStiffness(self, k):
        """
        Replace measured acceleration by mock one, a = k * self.pos/self.m
        """
        self.acc = k * self.x0.nominal_value / self.m * np.sin(self.omega_exc * self.t + self.xphase.nominal_value) + self.acc_errors
        
    def addMockWire(self, k, Q):
        """
        Add a mock wire's signal, of wire (k,Q). Useful to check if we can grab it, depending on its Q
        """
        T = 300. #s, period of the excitation signal
        timeOffset = T / (2. * np.pi * Q)
        print("---> addMockWire: offset = " + str(timeOffset) + " s")
        phase = 1./Q
        s = k * self.x0.nominal_value / self.m * np.sin(self.omega_exc * self.t + self.xphase.nominal_value - phase)
        self.acc += s

    def showFFT(self):
        dt = self.t[1] - self.t[0]
        n = np.size(self.t)
        freqs = fftfreq(n, dt)
        xfft = fft(self.pos - np.mean(self.pos))
        xnorm = np.abs(xfft)
        xphase = np.arctan2(np.imag(xfft), np.real(xfft))
        afft = fft(self.acc - np.mean(self.acc))
        anorm = np.abs(afft)
        aphase = np.arctan2(np.imag(afft), np.real(afft))

        xnorm = xnorm[freqs > 0]
        xphase = xphase[freqs > 0]
        anorm = anorm[freqs > 0]
        aphase = aphase[freqs > 0]
        freqs = freqs[freqs > 0]
        
        ax = plt.subplot(211)
        plt.plot(freqs, xnorm, label = 'Position')
        plt.plot(freqs, anorm, label = 'Acceleration')
        plt.legend()
        plt.yscale('log')
        plt.ylabel('|FFT|')
        plt.subplot(212, sharex = ax)
        plt.plot(freqs, xphase)
        plt.plot(freqs, aphase)
        plt.ylabel('Phase')
        plt.xlabel('f [Hz]')
        plt.xscale('log')
        plt.show()
        
        
    def getErrors(self, plotit = False):
        """
        Get measurement errors on acceleration (resampled at 1 Hz). Just fit twosuccessive sines and take residuals as errors.
        """
        freq = 2*np.pi / 300 #rad/s
        fit = pe.fitSingleSine(self.t, self.acc, freq_range = [1./500,1./200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit, silent = True)
        fit = pe.fitSingleSine(self.t, fit['residuals'], freq_range = [1./500,1./200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit, silent = True)
        self.acc_errors = fit['residuals']
        #self.acc_errors = getErrors(self.t, self.acc, plotit = plotit)

    def getExcitationCharacteristics(self, fitFrequency = False, plotit = False):
        """
        Get characteristics of position excitation: amplitude, frequency and phase
        """
        if not fitFrequency:
            freq = self.omega_exc
            #freq = 2*np.pi / 300 #Hz
        else:
            freq = None
        fit = pe.fitSingleSine(self.t, self.pos, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
        self.x0 = ufloat(fit['amplitude'], fit['err_amplitude'])
        self.xmean = ufloat(fit['mean'], fit['err_mean'])
        self.xphase = ufloat(fit['phase'], fit['err_phase'])
        self.xfreq = ufloat(fit['frequency'], 0) / (2.*np.pi)
        #print("---> getExcitationCharacteristics")
        #print('x0:', self.x0)
        ##print(self.xmean)
        #print('phase:', self.xphase)
        #print('freq:', self.xfreq)
        #print('red. chisq:', fit['rchi2'])
        if plotit:
            plt.plot(self.t, self.pos, label = 'Data')
            plt.plot(fit['x'], fit['fit'], label = 'Fit')
            plt.plot(fit['x'], fit['residuals'], label = 'Residuals')
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel('pos [m]')
            plt.show(block = True)

    def getAccelerationSine(self, fitFrequency = False, plotit = False, correct4exc = False, usePositionPhase = False):
        if correct4exc:
            acc = self.acc - self.omega_exc**2 * self.pos
        else:
            acc = self.acc
        if not fitFrequency:
            freq = 2*np.pi / 300 #Hz
        else:
            freq = None
        if usePositionPhase:
            phase = self.xphase.nominal_value
        else:
            phase = None
        fit = pe.fitSingleSine(self.t, acc, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = phase, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
        self.a0 = ufloat(fit['amplitude'], fit['err_amplitude'])
        self.b_0 = ufloat(fit['mean'], fit['err_mean'])
        self.aphase = ufloat(fit['phase'], fit['err_phase'])
        self.afreq = ufloat(fit['frequency'], 0) / (2.*np.pi)

        if plotit:
            plt.plot(self.t, self.acc, label = 'Data')
            plt.plot(fit['x'], fit['fit'], label = 'Fit')
            plt.plot(fit['x'], fit['residuals'], label = 'Residuals')
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel('acc [m]')
            plt.show(block = True)
            
        return acc

    

class lsq(suk):
    def __init__(self, _su, _is, _axis, mode, correct4theory_kE = False, e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 1e-4, phase = 'position', plotData = False):
        super(lsq, self).__init__(_su, _is, _axis, mode, correct4theory_kE = correct4theory_kE, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error, plotData = plotData)

    def fitModel(self, fit_kw = True, fit_lambda = False, kQ = None, Q = ufloat(100, 50), phase = 'position', taylorOrder = 4, recursive = True, verbose = True, plotit = False, full_output = False):
        """
        Overall least-square fit
        Q -- guess on wire's quality factor, or Q estimated from other axes
        """
        if taylorOrder is None:
                #Does not work with ufloat (Q)...
                taylorOrder = 4
        if fit_kw and fit_lambda:
            raise ValueError('Cannot fit kw and lambda simultaneously')
        if not fit_kw and not fit_lambda:
            fit_kw = True
        if phase == 'position':
            psi = self.xphase.nominal_value
        elif phase == 'acceleration':
            psi = self.aphase.nominal_value
        else:
            raise ValueError("Bad phase type")
        A = lsqMatrixModel(self.t, self.omega_exc, psi, phase)
        coeffs, cov, rspace, fspace, chi2, rchi2, ci, semiAnalCovParams = pe.lsq(self.t, self.acc, func = 'local model', empirical_model = A, svd = False, rms = np.std(self.acc, ddof = 1), cmp_ci = False, plotit = False, fourier = False, covMatrix = None)
        
        if recursive:
            res = rspace['residuals']
            rms = np.std(res, ddof = 1)
            coeffs, cov, rspace, fspace, chi2, rchi2, ci, semiAnalCovParams = pe.lsq(self.t, self.acc, func = 'local model', empirical_model = A, svd = False, rms = rms, cmp_ci = False, plotit = False, fourier = False, covMatrix = None)
        err = np.sqrt(np.diag(cov))

        #b_est = coeffs[0]
        #a0_est = coeffs[1]
        #aw_est = coeffs[2]
        b_est = ufloat(coeffs[0], err[0])
        a0_est = ufloat(coeffs[1], err[1])
        aw_est = ufloat(coeffs[2], err[2])
        if verbose:
            print('---> lsq.fitModel')
            print('   estimates')
            print('     ', b_est, a0_est, aw_est)
            print('     ', err)
            print('   covariance')
            print(cov)
            print('   rchi2', rchi2)

        if plotit:
            ax = plt.subplot(311)
            plt.plot(self.t, self.pos)
            plt.ylabel('Pos [m]')
            plt.subplot(312, sharex = ax)
            plt.plot(self.t, self.acc, label = 'data')
            plt.plot(self.t, rspace['fit'], label = 'fit')
            plt.plot(self.t, rspace['residuals'], label = 'residuals')
            plt.legend()
            plt.ylabel('Acc [m/s^2]')
            plt.subplot(313, sharex = ax)
            plt.plot(self.t, rspace['residuals'])
            plt.ylabel('Residuals')
            plt.xlabel('t [s]')
            plt.show()

        #phi = 1./Q
        if fit_kw:
            if taylorOrder is None:
                raise NotImplementedError('Does not work with ufloat (Q)...')
                #k_w = -6 * self.m / self.x0 * aw_est / np.sin(1./Q)
                #k_0 = self.m / self.x0 * (a0_est + aw_est * np.cos(1./Q) / np.sin(1./Q))
            elif taylorOrder == 4:
                k_w = -6 * self.m / self.x0 * Q**3 / (6 * Q**2 - 1) * aw_est
                k_0 = self.m / self.x0 * (a0_est + aw_est * (1 - 12 * Q**2 + 24 * Q**4) / (4 * Q * (6 * Q**2 -1)))
            elif taylorOrder == 2:
                k_w = -self.m / self.x0 * aw_est * Q
                k_0 = self.m / self.x0 * (a0_est + aw_est * (2 * Q**2 - 1) / (2.*Q))
            else:
                raise NotImplementedError("Computed only for order 2 and 4 (or no Taylor-expansion whatsoever)!")
            lambda_v = ufloat(0,0)
        elif fit_lambda:
            if kQ is None:
                raise ValueError('To estimate lambda, k/Q must be set')
            k_w = kQ * Q
            k_0 = self.m / self.x0 * a0_est - kQ * (24 * Q**4 - 12 * Q**2 + 1) / (24 * Q**3)
            lambda_v = self.m / (self.x0 * self.omega_exc) * aw_est + kQ / self.omega_exc * (2 * Q**2 - 1) / (2 * Q**2)
        else:
            raise ValueError('WTH?!?')
        self.b_est = b_est
        self.k_0 = k_0
        self.k_w = k_w

        if full_output:
            _dic = {'b': b_est, 'a0': a0_est, 'aw': aw_est, 'k_0': k_0, 'k_w': k_w, 'Q': Q, 'lambda_v': lambda_v, 'rchi2': rchi2}
            return b_est, k_0, k_w, _dic
        else:
            if fit_kw:
                return b_est, k_0, k_w
            else:
                return b_est, k_0, lambda_v
            
    def subtractTheory(self, k_0, verbose = True, plotit = False):
    #def subtractTheory(self, k_0, Q = ufloat(100, 50), taylorOrder = 4, phase = 'position', recursive = True, verbose = True, plotit = False):
        #if k_0 is None and self.k_0 is None:
        #    raise NotImplementedError('Better not to have to do it again...')
        #    b_est, k_0, k_w = self.fitModel(Q = Q, taylorOrder = taylorOrder, phase = phase, recursive = recursive, verbose = verbose, plotit = plotit)
        kE_th = self.getTheoryElectrostaticStiffness()
        kN = self.getNewtonStiffness()
        k_corr = k_0 - kE_th - kN - self.omega_exc**2 * self.m
        print('---> subtractTheory')
        print('   k0', self.k_0)
        print('   kE_th', kE_th)
        print('   kN', kN)
        print('   kexc', self.omega_exc**2 * self.m)
        return k_corr, kE_th, kN
        
    def fit_eStiffness(self, correct4exc = True, usePositionPhase = False, plotit = False):
        """
        Compute naive stiffness as ratio between acceleration and position maxima
        """
        #self.getExcitationCharacteristics()
        acc = self.getAccelerationSine(correct4exc = correct4exc, usePositionPhase = usePositionPhase, plotit = plotit)
        k = (self.a0 / self.x0) * self.m
        if self._axis in ['Y', 'Z'] and not usePositionPhase:
            #the minus sign is here b/c we know that position and acceleration are out of phase, but only for radial axes
            k *= -1
        #print('---> fit_eStiffness', k)
        return k

    def fit_egwStiffness(self, forceFreq = True, correct4exc = True, correct4viscousDamping = False, plotit = True):
        """
        Compute naive electrostatic and gold wire stiffness as ratio between acceleration and position maxima
        Gold wire stiffness is computed using the residuals from the electrostatic stiffness, which are non-zero is there's an out-of-phase sine looming in the data (which can be due to a poor-Q wire)
        """
        acc = self.getAccelerationSine(correct4exc = correct4exc)
        kE = self.fit_eStiffness(correct4exc = correct4exc)
        #eContribution = kE.nominal_value / self.m * self.pos + self.b_0.nominal_value #this is indeed +k here because acceleration is the control acceleration, for which we have a = +k/m*x...
        eContribution = -kE.nominal_value / self.m * self.x0.nominal_value * np.sin(self.omega_exc * self.t + self.aphase.nominal_value) + self.b_0.nominal_value

        if correct4viscousDamping:
            lmbda_gw, lmbda_wk, gwvd_acc = self.fit_gwViscousDamping(plotit = plotit)
            acc -= gwvd_acc
        else:
            lmbda_gw = 0
            lmbda_wk = 0

        #residual, to which we can think gold wire contribute, especially if outof phase with position
        if forceFreq:
            freq = self.omega_exc
            #freq = 2. * np.pi / 300 #Hz
        else:
            freq = None
        res = acc - eContribution
        resFit = pe.fitSingleSine(self.t, res, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
        gw_amp = ufloat(resFit['amplitude'], resFit['err_amplitude'])
        gw_phase = resFit['phase']
        gw_e_deltaPhase = resFit['phase'] - self.xphase.nominal_value
        gw_Q = 1./np.abs(gw_e_deltaPhase)
        gw_k = (gw_amp / self.x0) * self.m
        print("--> fit_egwStiffness")
        print("   electro stiffness", kE)
        print("   GW stiffness", gw_k)
        print("   GW Q", gw_Q)
        print("   GW k/Q", gw_k / gw_Q)
        print("      Phase difference [rad]", gw_e_deltaPhase)
        if not forceFreq:
            print(" (Fitted) freq --expect ", 1./300)
            print(" freq [Hz]", resFit['frequency'] / (2. * np.pi))
            print(" period [s]", (2. * np.pi)/resFit['frequency'])
        
        if plotit:
            ax = plt.subplot(311)
            plt.plot(self.t, self.acc, label = 'Data')
            if correct4exc:
                plt.plot(self.t, acc, label = 'Data - exc')
            plt.plot(self.t, eContribution, label = 'Electro-k')
            plt.plot(self.t, res, label = 'Data - Electro-k')
            plt.legend()
            plt.ylabel('Acc')
            plt.subplot(312, sharex = ax)
            plt.plot(self.t, res, label = 'GW')
            plt.plot(self.t, resFit['fit'], label = 'GW fit')
            plt.legend()
            plt.ylabel('GW')
            plt.subplot(313, sharex = ax)
            plt.plot(self.t, res - resFit['fit'])
            plt.ylabel('Overall residuals')
            plt.xlabel('t [s]')
            plt.show(block = True)

        return kE, gw_k, gw_Q, gw_k / gw_Q, gw_e_deltaPhase, lmbda_wk, res - resFit['fit']

    def fit_gwViscousDamping(self, plotit = False):
        phase = self.xphase.nominal_value + np.pi / 2
        freq = self.omega_exc
        acc = self.getAccelerationSine(correct4exc = False)
        kE = self.fit_eStiffness(correct4exc = False)
        eContribution = kE.nominal_value / self.m * self.pos + self.b_0.nominal_value #this is indeed +k here because acceleration is the control acceleration, for which we have a = +k/m*x...

        #residual, to which we can think gold wire contribute, especially if outof phase with position
        res = acc - eContribution
        resFit = pe.fitSingleSine(self.t, res, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = phase, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
        lmbda_w = ufloat(resFit['amplitude'], resFit['err_amplitude']) / self.omega_exc * self.m
        acc_amp = ufloat(resFit['amplitude'], resFit['err_amplitude'])
        lmbda_w = acc_amp / (self.omega_exc * self.x0) * self.m
        lmbda_wk = lmbda_w * self.omega_exc #related (out-of-phase) stiffness
        print("--> fit_gwViscousDamping")
        print("   lmbda_gw [Ns/m]", lmbda_w)
        print("   lmbda_gwk [N/m]", lmbda_wk)

        if plotit:
            ax = plt.subplot(311)
            plt.plot(self.t, self.acc, label = 'Data')
            plt.plot(self.t, eContribution, label = 'Electro-k')
            plt.plot(self.t, res, label = 'Data - Electro-k')
            plt.legend()
            plt.ylabel('Acc')
            plt.subplot(312, sharex = ax)
            plt.plot(self.t, res, label = 'GW')
            plt.plot(self.t, resFit['fit'], label = 'GW viscous damping fit')
            plt.legend()
            plt.ylabel('GW')
            plt.subplot(313, sharex = ax)
            plt.plot(self.t, res - resFit['fit'])
            plt.ylabel('Overall residuals')
            plt.xlabel('t [s]')
            plt.show(block = True)
        
        return lmbda_w, lmbda_wk, resFit['fit']

    def fit_residualSine(self, plotit = False):
        """
        Look for a lingering sine in data after excitation, electrostatic stiffness (including gravity and gold wire if Q>>1), and gold wire's viscous and internal damping/stiffness have been corrected
        """
        kE, gw_k, gw_Q, kQ, dp, lvdgw, res = self.fit_egwStiffness(forceFreq = True, correct4exc = True, correct4viscousDamping = True, plotit = False)
        resFit = pe.fitSingleSine(self.t, res, freq_range = [2*np.pi/1500,2*np.pi/20], nfreq = 500, freq = None, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
        amp = ufloat(resFit['amplitude'], resFit['err_amplitude'])
        freq = resFit['frequency'] / (2. * np.pi)
        phase = resFit['phase']
        print("---> fit_residualSine")
        print("  amplitude", amp)
        print("  freq [Hz]", freq)
        print("  phase [rad]", phase)

        if plotit:
            ax = plt.subplot(211)
            plt.plot(self.t, res)
            plt.plot(self.t, resFit['fit'])
            plt.ylabel('Residual acceleration')
            plt.subplot(212, sharex = ax)
            plt.plot(self.t, res - resFit['fit'])
            plt.ylabel('Fit residuals')
            plt.xlabel('t [s]')
            plt.show(block = True)

        return amp, freq, phase



class mcmc(suk):
    def __init__(self, _su, _is, _axis, mode, modelType = 'stiffness', correct4theory_kE = False, e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 1e-4, plotData = False):
        super(mcmc, self).__init__(_su, _is, _axis, mode, correct4theory_kE = correct4theory_kE, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error, plotData = plotData)
        self.model = modelType
        if self.model == 'Yukawa':
            self.nparams = 6 #alpha, lmbda, kE, kGW, QGW, b
        elif self.model == 'stiffness':
            self.nparams = 4 #k0 (=kexc+kE+kgrav), kw, Q, b
        elif self.model == 'stiffness+damping':
            self.nparams = 5 #k0 (=kexc+kE+kgrav), kw, Q, Lv, b
        else:
            raise NotImplementedError()
        self.nwalkers = 32
        
    def initMaxL(self, init_error = 0.):
        if self.model == 'Yukawa':
            zero = np.array([alpha_0, lmbda_0, self.kE_0, kGW_0, QGW_0, self.b_0.nominal_value])
            #initial = zero * (1 + init_error * np.random.rand(self.nparams))
        elif self.model == 'stiffness':
            kexc = self.m * self.omega_exc**2
            kN = self.getNewtonStiffness()
            k0 = self.kE_0 + kexc + kN
            zero = np.array([k0, kGW_0, QGW_0, self.b_0.nominal_value])
        elif self.model == 'stiffness+damping':
            kexc = self.m * self.omega_exc**2
            kN = self.getNewtonStiffness()
            k0 = self.kE_0 + kexc + kN
            zero = np.array([k0, kGW_0, QGW_0, Lv_0, self.b_0.nominal_value])
        else:
            raise NotImplementedError()
        initial = zero * (1 + init_error * np.random.rand(self.nparams))
        return initial
        
    def maxL(self, init_error = 0., printit = False, plotit = False):
        """
        log_likelihood() signature: log_likelihood(theta, x, y, yerr, _is = None, _su = None, _axis = None, correct4theory_kE = False)
        """
        nll = lambda *args: -log_likelihood(self.model, *args)
        initial = self.initMaxL(init_error = init_error)
        if self.model == 'Yukawa':
            #params = alpha, lmbda, self.kE, kGW, QGW, b
            bnds = ((None, None), (0, None), (self.kE_bounds[0], self.kE_bounds[1]), (kGW_range[0], kGW_range[1]), (QGW_range[0], QGW_range[1]), (self.b_0.nominal_value - 5 * self.b_0.std_dev, self.b_0.nominal_value + 5 * self.b_0.std_dev))
        elif self.model == 'stiffness':
            #params = kE+kexc+kN, kGW, QGW, b
            bnds = ((self.kE_bounds[0], self.kE_bounds[1]),  (kGW_range[0], kGW_range[1]), (QGW_range[0], QGW_range[1]), (self.b_0.nominal_value - 5 * self.b_0.std_dev, self.b_0.nominal_value + 5 * self.b_0.std_dev)) #kE is so largely dominant that we can use its bounds for k0
        elif self.model == 'stiffness+damping':
            #params = kE+kexc+kN, kGW, QGW, b
            bnds = ((self.kE_bounds[0], self.kE_bounds[1]),  (kGW_range[0], kGW_range[1]), (QGW_range[0], QGW_range[1]), (Lv_range[0], Lv_range[1]), (self.b_0.nominal_value - 5 * self.b_0.std_dev, self.b_0.nominal_value + 5 * self.b_0.std_dev)) #kE is so largely dominant that we can use its bounds for k0
        else:
            raise NotImplementedError()
        #x = self.pos
        x = self.t
        y = self.acc
        yerr = self.acc_errors
        soln = minimize(nll, initial, args = (x, y, yerr, self._is, self._su, self._axis, self.xphase.nominal_value, self.correct4theory_kE), bounds = bnds)

        if self.model == 'Yukawa':
            alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml = soln.x
        elif self.model == 'stiffness':
            kE_ml, kGW_ml, QGW_ml, a0_ml = soln.x #careful. Despite its name, kE also contains excitation and gravity
        elif self.model == 'stiffness+damping':
            kE_ml, kGW_ml, QGW_ml, Lv_ml, a0_ml = soln.x #careful. Despite its name, kE also contains excitation and gravity
        else:
            raise NotImplementedError()

        if printit:
            print()
            print('\n---> ML estimates')
            print('kE = ' + str(kE_ml))
            print('kGW = ' + str(kGW_ml))
            print('QGW = ' + str(QGW_ml))
            print('b = ' + str(a0_ml))
            if self.model == 'stiffness+damping':
                print('Lv = ' + str(Lv_ml))
            if self.model == 'Yukawa':
                print('alpha = ' + str(alpha_ml))
                print('lambda = ' + str(lmbda_ml))
            print()

        if plotit:
            if self.model == 'Yukawa':
                ml = funY(self._is, self._su, self._axis, self.t, alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml, psi = self.xphase.nominal_value, correct4theory_kE = self.correct4theory_kE)
            elif self.model == 'stiffness':
                ml = funK(self._is, self._su, self._axis, self.t, kE_ml, kGW_ml, QGW_ml, a0_ml, psi = self.xphase.nominal_value)
            elif self.model == 'stiffness+damping':
                ml = funKv(self._is, self._su, self._axis, self.t, kE_ml, kGW_ml, QGW_ml, Lv_ml, a0_ml, psi = self.xphase.nominal_value)
            else:
                raise NotImplementedError()
            plt.errorbar(x, y, yerr = yerr, marker = '.', linestyle = '', color = 'black', label = 'Data')
            plt.plot(x, ml, color = 'blue', label = 'ML')
            plt.legend(loc = 'upper right')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.show(block = True)

        if self.model == 'Yukawa':
            return alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml, initial, soln
        elif self.model == 'stiffness':
            return kE_ml, kGW_ml, QGW_ml, a0_ml, initial, soln
        elif self.model == 'stiffness+damping':
            return kE_ml, kGW_ml, QGW_ml, Lv_ml, a0_ml, initial, soln
        else:
            raise NotImplementedError()
            
    
    def log_prior(self, theta):
        if self.model == 'Yukawa':
            alpha, lmbda, kE, kGW, QGW, b = theta
            alpha_min = alpha_range[0]
            alpha_max = alpha_range[1]
            lmbda_min = lmbda_range[0]
            lmbda_max = lmbda_range[1]
            #kE_min = self.kE_0 - 5 * self.kE_rms0
            #kE_max = self.kE_0 + 5 * self.kE_rms0
            kE_min = -5 * np.abs(self.kE_0)
            kE_max = 5 * np.abs(self.kE_0)
            kGW_min = kGW_range[0]
            kGW_max = kGW_range[1]
            QGW_min = QGW_range[0]
            QGW_max = QGW_range[1]
            #a0_min = self.b_0.nominal_value - 15 * self.b_0.std_dev
            #a0_max = self.b_0.nominal_value + 15 * self.b_0.std_dev
            a0_min = -2 * np.abs(self.b_0.nominal_value)
            a0_max = 2 * np.abs(self.b_0.nominal_value)
        
            if alpha_min < alpha < alpha_max and lmbda_min < lmbda < lmbda_max and kE_min < kE < kE_max and kGW_min < kGW < kGW_max and QGW_min < QGW < QGW_max and a0_min < b < a0_max:
                return 0.
        elif self.model == 'stiffness':
            kE, kGW, QGW, b = theta
            #kE_min = self.kE_0 - 15 * self.kE_rms0
            #kE_max = self.kE_0 + 15 * self.kE_rms0
            kE_min = -5 * np.abs(self.kE_0)
            kE_max = 5 * np.abs(self.kE_0)
            kGW_min = kGW_range[0]
            kGW_max = kGW_range[1]
            QGW_min = QGW_range[0]
            QGW_max = QGW_range[1]
            #a0_min = self.b_0.nominal_value - 15 * self.b_0.std_dev
            #a0_max = self.b_0.nominal_value + 15 * self.b_0.std_dev
            a0_min = -2 * np.abs(self.b_0.nominal_value)
            a0_max = 2 * np.abs(self.b_0.nominal_value)
        
            if kE_min < kE < kE_max and kGW_min < kGW < kGW_max and QGW_min < QGW < QGW_max and a0_min < b < a0_max:
                return 0.
        elif self.model == 'stiffness+damping':
            kE, kGW, QGW, Lv, b = theta
            #kE_min = self.kE_0 - 15 * self.kE_rms0
            #kE_max = self.kE_0 + 15 * self.kE_rms0
            kE_min = -5 * np.abs(self.kE_0)
            kE_max = 5 * np.abs(self.kE_0)
            kGW_min = kGW_range[0]
            kGW_max = kGW_range[1]
            QGW_min = QGW_range[0]
            QGW_max = QGW_range[1]
            #a0_min = self.b_0.nominal_value - 15 * self.b_0.std_dev
            #a0_max = self.b_0.nominal_value + 15 * self.b_0.std_dev
            a0_min = -2 * np.abs(self.b_0.nominal_value)
            a0_max = 2 * np.abs(self.b_0.nominal_value)
            Lv_min = Lv_range[0]
            Lv_max = Lv_range[1]
        
            if kE_min < kE < kE_max and kGW_min < kGW < kGW_max and QGW_min < QGW < QGW_max and a0_min < b < a0_max and Lv_min < Lv < Lv_max:
                return 0.
        else:
            raise NotImplementedError()
        return -np.inf

    def log_probability(self, theta, x, y, yerr):
        """
        log_likelihood signature: log_likelihood(theta, x, y, yerr, _is = None, _su = None, _axis = None, psi = 0, correct4theory_kE = False)
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(self.model, theta, x, y, yerr, _is = self._is, _su = self._su, _axis = self._axis, psi = self.xphase.nominal_value, correct4theory_kE = self.correct4theory_kE)


    def mcmc(self, init_error = 0., nsteps = 5000):
        x = self.t
        y = self.acc
        yerr = self.acc_errors
        maxL_out = self.maxL(init_error = init_error)
        if self.model == 'stiffness':
            k0_ml, kw_ml, Q_ml, a0_ml, initial, soln = maxL_out
        elif self.model == 'stiffness+damping':
            k0_ml, kw_ml, Q_ml, Lv_ml, a0_ml, initial, soln = maxL_out
        elif self.model == 'Yukawa':
            alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml, initial, soln = maxL_out
        else:
            raise NotImplementedError()
        pos = soln.x * (1 + init_error * np.random.rand(self.nwalkers, self.nparams))
        nwalkers, ndim = pos.shape
        #print("\nCochonne MCMC!")
        #print(alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml)
        #print(self.log_prior(soln.x))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args = (x, y, yerr))
        sampler.run_mcmc(pos, nsteps, progress = True)

        tau = sampler.get_autocorr_time(quiet = True)
        print('MCMC autocorr time', tau)
        
        self.sampler = sampler

    
    def plotWalkers(self):
        """
        sampler -- mcmc output
        """
        fig, axes = plt.subplots(self.nparams, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        if self.model == 'Yukawa':
            labels = ["alpha", "lambda", "kE", "kGW", "Q", "b"]
        elif self.model == 'stiffness':
            labels = ["k0", "kGW", "Q", "b"]
        elif self.model == 'stiffness+damping':
            labels = ["k0", "kGW", "Q", "Lv", "b"]
        else:
            raise NotImplementedError()
        for i in range(self.nparams):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        plt.show(block = True)

    def plotPosterior(self):
        if self.model == 'Yukawa':
            labels = ["alpha", "lambda", "kE", "kGW", "Q", "b"]
        elif self.model == 'stiffness':
            labels = ["k0", "kGW", "Q", "b"]
        elif self.model == 'stiffness+damping':
            labels = ["k0", "kGW", "Q", "Lv", "b"]
        else:
            raise NotImplementedError()
        discard=500
        flat_samples = self.sampler.get_chain(discard=discard, thin=15, flat=True)
        print(flat_samples.shape)

        for i in range(self.nparams):
            res = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(res)
            print(labels[i], res[1], q[0], q[1])
    
        fig = corner.corner(flat_samples, labels=labels)
        plt.show(block = True)
    


def naive_kfit(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM', usePositionPhase = False, correct4exc = False, plotit = True):
    """
    Fit total stiffness a la Ratana
    """
    s = lsq(_su, _is, _axis, mode)
    k = s.fit_eStiffness(correct4exc = correct4exc, usePositionPhase = usePositionPhase, plotit = plotit)
    print(s.xphase)
    print(s.aphase)

def k_fit(_su = 'SUEP', _is = 'IS1', _axis = 'X', mode = 'HRM', fit_kw = True, fit_lambda = False, kQ = ufloat(1.1e-3, 0.2e-3), Q = ufloat(1, 50), taylorOrder = 4, phase = 'position', e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 1e-4, plotit = False, mock = {'on': False, 'k0': -1.5e-2, 'kw': 1e-3, 'Q': 50}):
    """
    Least-square fit of overall model
    """
    s = lsq(_su, _is, _axis, mode, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error)
    if mock['on']:
        if not 'k0' in mock:
            mock['k0'] = -1.5e-2
        if not 'kw' in mock:
            mock['kw'] = 1e-3
        if not 'Q' in mock:
            mock['Q'] = 50
            Q = ufloat(mock['Q'], 0.3 * mock['Q'])
        s.mkMock(mock['k0'], mock['kw'], mock['Q'])
    b, k0, kw, _dic = s.fitModel(fit_kw = fit_kw, fit_lambda = fit_lambda, kQ = kQ, Q = Q, taylorOrder = taylorOrder, phase = phase, recursive = True, plotit = plotit, full_output = True)
    #print(b, k0, kw, kw / Q)
    if mock['on']:
        print('Input ks')
        print('  k0', mock['k0'])
        print('  kw', mock['kw'])
        print('  kw/Q', mock['kw'] / mock['Q'])

    k_corr, kE_th, kN = s.subtractTheory(s.k_0)
    #print(k_corr, kE_th, kN)
    res = {'k0': _dic['k_0'], 'kw': _dic['k_w'], 'Q': _dic['Q'], 'lambda_v': _dic['lambda_v'], 'k_corr': k_corr, 'kE_th': kE_th, 'kN': kN}
    return res
#return k0, kw, kw / Q, k_corr, kE_th, kN

def k_fit_radial(_su = 'SUEP', _is = 'IS1', kyQ = ufloat(0.3e-3,0.5e-3), kzQ = ufloat(1.3e-3,2e-3), taylorOrder = 4, phase = 'position', e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 0, Qtrig = True, plotit = False):
    """
    Compute radial stiffnesses by assuming cylindrical geometry for electrostatic stiffness
    """
    #should be something like that...
    if Qtrig:
        taylorOrder = None
    #... but I'm too lazy to do it now, so cope with Taylor expansion!
    #if taylorOrder is None:
    #    taylorOrder = 4
        
    sYHRM = lsq(_su, _is, 'Y', 'HRM', e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error)
    b, _dummyk0, _dummykw, _dicYH = sYHRM.fitModel(Q = ufloat(1,0), taylorOrder = taylorOrder, phase = phase, recursive = True, plotit = plotit, full_output = True)
    sYFRM = lsq(_su, _is, 'Y', 'FRM', e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error)
    b, _dummyk0, _dummykw, _dicYF = sYFRM.fitModel(Q = ufloat(1,0), taylorOrder = taylorOrder, phase = phase, recursive = True, plotit = plotit, full_output = True)
    sZHRM = lsq(_su, _is, 'Z', 'HRM', e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error)
    b, _dummyk0, _dummykw, _dicZH = sZHRM.fitModel(Q = ufloat(1,0), taylorOrder = taylorOrder, phase = phase, recursive = True, plotit = plotit, full_output = True)
    sZFRM = lsq(_su, _is, 'Z', 'FRM', e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error)
    b, _dummyk0, _dummykw, _dicZF = sZFRM.fitModel(Q = ufloat(1,0), taylorOrder = taylorOrder, phase = phase, recursive = True, plotit = plotit, full_output = True)

    a0_YF = _dicYF['a0']
    aw_YF = _dicYF['aw']
    a0_YH = _dicYH['a0']
    aw_YH = _dicYH['aw']
    a0_ZF = _dicZF['a0']
    aw_ZF = _dicZF['aw']
    a0_ZH = _dicZH['a0']
    aw_ZH = _dicZH['aw']
    chiY = kyQ
    chiZ = kzQ

    print('---> k_fit_radial, estimated params')
    print('  a0_YF', a0_YF)
    print('  a0_ZF', a0_ZF)
    print('  a0_YH', a0_YH)
    print('  a0_ZH', a0_ZH)
    print('  aw_YF', aw_YF)
    print('  aw_ZF', aw_ZF)
    print('  aw_YH', aw_YH)
    print('  aw_ZH', aw_ZH)
    
    x0 = sYHRM.x0.nominal_value
    m = sYHRM.m
    A = 2 * (chiY - chiZ) * x0 / m
    B = a0_YF + a0_YH - a0_ZF - a0_ZH
    print('B/A...', B/A)
    quartic_coeff = B/A
    Qs = solve4Q_mc(quartic_coeff, Qstart = 0.1, Qend = 1000, step = 0.5, trig = Qtrig)
    print(Qs)
    #keep the highest Q. No special reason for that, but no better choice...
    Q = Qs[-1]
    #...unless Qtrig, in which case there's only one solution
    if Qtrig:
        Q = Qs[0]

    print("   quality factor Q", Q)

    #real parameters
    if Qtrig:
        #does not work with Q being a ufloat...
        Qtrig = False
    if Qtrig:
        #does not work with Q being a ufloat...
        raise NotImplementedError("does not work with Q being a ufloat...")
        Qcos = Q * np.cos(1./Q)
        Qsin = Q * np.sin(1./Q)
    else:
        #Taylor-expansion of order 4
        Qcos = (24 * Q**4 - 12 * Q**2 + 1) / (24 * Q**3)
        Qsin = (2 * Q**2 - 1) / (2 * Q**2)
    kwy = chiY * Q
    kwz = chiZ * Q
    kappa_0F_y = a0_YF - x0 / m * chiY * Qcos
    kappa_0F_z = a0_ZF - x0 / m * chiZ * Qcos
    kappa_0H_y = a0_YH - x0 / m * chiY * Qcos
    kappa_0H_z = a0_ZH - x0 / m * chiZ * Qcos
    kappa_LyF = aw_YF + x0 / m * chiY * Qsin
    kappa_LyH = aw_YH + x0 / m * chiY * Qsin
    kappa_LzF = aw_ZF + x0 / m * chiZ * Qsin
    kappa_LzH = aw_ZH + x0 / m * chiZ * Qsin
    #July version... misses Q**2 at denominator... but not a problem since Q~1
    #kappa_0F_y = a0_YF - x0 / m * chiY * (24 * Q**4 - 12 * Q**2 + 1) / (24 * Q**3)
    #kappa_0F_z = a0_ZF - x0 / m * chiZ * (24 * Q**4 - 12 * Q**2 + 1) / (24 * Q**3)
    #kappa_0H_y = a0_YH - x0 / m * chiY * (24 * Q**4 - 12 * Q**2 + 1) / (24 * Q**3)
    #kappa_0H_z = a0_ZH - x0 / m * chiZ * (24 * Q**4 - 12 * Q**2 + 1) / (24 * Q**3)
    #kappa_LyF = aw_YF + x0 / m * chiY * (2 * Q**2 - 1) / (2 * Q)
    #kappa_LyH = aw_YH + x0 / m * chiY * (2 * Q**2 - 1) / (2 * Q)
    #kappa_LzF = aw_ZF + x0 / m * chiZ * (2 * Q**2 - 1) / (2 * Q)
    #kappa_LzH = aw_ZH + x0 / m * chiZ * (2 * Q**2 - 1) / (2 * Q)

    k0_yF = m / x0 * kappa_0F_y
    k0_zF = m / x0 * kappa_0F_z
    k0_yH = m / x0 * kappa_0H_y
    k0_zH = m / x0 * kappa_0H_z
    lambda_yF = m / (x0 * sYHRM.omega_exc) * kappa_LyF
    lambda_zF = m / (x0 * sYHRM.omega_exc) * kappa_LzF
    lambda_yH = m / (x0 * sYHRM.omega_exc) * kappa_LyH
    lambda_zH = m / (x0 * sYHRM.omega_exc) * kappa_LzH

    #and corrected for theory
    k_corr_yH, kE_th_yH, kN = sYHRM.subtractTheory(k0_yH, verbose = True, plotit = False)
    k_corr_yF, kE_th_yF, kN = sYFRM.subtractTheory(k0_yF, verbose = True, plotit = False)
    k_corr_zH, kE_th_zH, kN = sZHRM.subtractTheory(k0_zH, verbose = True, plotit = False)
    k_corr_zF, kE_th_zF, kN = sZFRM.subtractTheory(k0_zF, verbose = True, plotit = False)
    #k_corr_yH, kE_th_yH, kN = sYHRM.subtractTheory(k0_yH, Q = Q, taylorOrder = 4, phase = 'position', recursive = True, verbose = True, plotit = False)
    #k_corr_yF, kE_th_yF, kN = sYFRM.subtractTheory(k0_yF, Q = Q, taylorOrder = 4, phase = 'position', recursive = True, verbose = True, plotit = False)
    #k_corr_zH, kE_th_zH, kN = sZHRM.subtractTheory(k0_zH, Q = Q, taylorOrder = 4, phase = 'position', recursive = True, verbose = True, plotit = False)
    #k_corr_zF, kE_th_zF, kN = sZFRM.subtractTheory(k0_zF, Q = Q, taylorOrder = 4, phase = 'position', recursive = True, verbose = True, plotit = False)
        
    res = {'k0_yFRM': k0_yF, 'k0_zFRM': k0_zF, 'k0_yHRM': k0_yH, 'k0_zHRM': k0_zH, 'kw_y': kwy, 'kw_z': kwz, 'kw_y/Q': kwy / Q, 'kw_z/Q': kwz / Q, 'Q': Q, 'lambda_yFRM': lambda_yF, 'lambda_zFRM': lambda_zF, 'lambda_yHRM': lambda_yH, 'lambda_zHRM': lambda_zH, 'kth_yHRM': kE_th_yH, 'kth_yFRM': kE_th_yF, 'kth_zHRM': kE_th_zH, 'kth_zFRM': kE_th_zF, 'k_corr_yHRM': k_corr_yH, 'k_corr_yFRM': k_corr_yF, 'k_corr_zHRM': k_corr_zH, 'k_corr_zFRM': k_corr_zF, 'kN': kN, 'omega': sYHRM.omega_exc, 'x0': x0, 'm': m}

    print("---> Instrumental parameters")
    print(" -----> Y axis / HRM")
    print("   k0 [N/m]", res['k0_yHRM'])
    print("   kw [N/m]", res['kw_y'])
    print("   kw / Q [N/m]", res['kw_y/Q'])
    print("   lambda [Ns/m]", res['lambda_yHRM'])
    print(" -----> Z axis / HRM")
    print("   k0 [N/m]", res['k0_zHRM'])
    print("   kw [N/m]", res['kw_z'])
    print("   kw / Q [N/m]", res['kw_z/Q'])
    print("   lambda [Ns/m]", res['lambda_zHRM'])
    print(" -----> Y axis / FRM")
    print("   k0 [N/m]", res['k0_yFRM'])
    print("   kw [N/m]", res['kw_y'])
    print("   kw / Q [N/m]", res['kw_y/Q'])
    print("   lambda [Ns/m]", res['lambda_yFRM'])
    print(" -----> Z axis / FRM")
    print("   k0 [N/m]", res['k0_zFRM'])
    print("   kw [N/m]", res['kw_z'])
    print("   kw / Q [N/m]", res['kw_z/Q'])
    print("   lambda [Ns/m]", res['lambda_zFRM'])

    print("---> Corrected electrostatic stiffness")
    print(" -----> Y axis / HRM")
    print("   kE [N/m]", res['k_corr_yHRM'])
    print(" -----> Z axis / HRM")
    print("   kE [N/m]", res['k_corr_zHRM'])
    print(" -----> Y axis / FRM")
    print("   kE [N/m]", res['k_corr_yFRM'])
    print(" -----> Z axis / FRM")
    print("   kE [N/m]", res['k_corr_zFRM'])
    
    return res
    

def plt_kvsQ(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM', Qerr = 20, taylorOrder = 4, phase = 'position', mock = {'on': False, 'k0': -1.5e-2, 'kw': 1e-3, 'Q': 50}, plotit = False):
    """
    Least-square fit of overall model
    """
    if mock['on']:
        if not 'k0' in mock:
            mock['k0'] = -1.5e-2
        if not 'kw' in mock:
            mock['kw'] = 1e-3
        if not 'Q' in mock:
            mock['Q'] = 50
            
    nQ = 50
    Qs = np.linspace(1, 100, nQ)
    kE_0 = np.zeros(nQ)
    kE_err = np.zeros(nQ)
    kw_0 = np.zeros(nQ)
    kw_err = np.zeros(nQ)
    kQ_0 = np.zeros(nQ)
    kQ_err = np.zeros(nQ)
    for i in range(nQ):
        k0, kw, kwQ, k_corr, kE_th, kN = k_fit(_su = _su, _is = _is, _axis = _axis, mode = mode, Q = ufloat(Qs[i], Qerr), taylorOrder = taylorOrder, phase = phase, mock = mock, plotit = plotit)
        kE_0[i] = k0.nominal_value
        kE_err[i] = k0.std_dev
        kw_0[i] = kw.nominal_value
        kw_err[i] = kw.std_dev
        kQ_0[i] = kwQ.nominal_value
        kQ_err[i] = kwQ.std_dev

    ax = plt.subplot(211)
    plt.errorbar(Qs, kE_0, yerr = kE_err, linestyle = '', marker = 'd', label = 'kE')
    plt.errorbar(Qs, kw_0, yerr = kw_err, linestyle = '', marker = 's', label = 'kw')
    plt.errorbar(Qs, kE_0 + kw_0, yerr = np.sqrt(kE_err**2 + kw_err**2), marker = 'o', label = 'kE+kw')
    if mock['on']:
        ym, yM = plt.ylim()
        plt.plot([mock['Q'], mock['Q']], [ym, yM], color = 'black')
        plt.plot([Qs[0], Qs[-1]], [mock['k0'], mock['k0']], label = 'Input kE')
        plt.plot([Qs[0], Qs[-1]], [mock['kw'], mock['kw']], label = 'Input kw')
    plt.legend()
    plt.ylabel('ks')
    plt.subplot(212, sharex = ax)
    plt.errorbar(Qs, kQ_0, yerr = kQ_err, linestyle = '', marker = '.', label = 'Estimated kw/Q', color = 'black')
    if mock['on']:
        plt.plot([Qs[0], Qs[-1]], [mock['kw']/mock['Q'], mock['kw']/mock['Q']], label = 'Input kw/Q', color = 'black')
    plt.legend()
    plt.ylabel('kw/Q')
    plt.xlabel('Prior Q')
    plt.show()



def k_fits(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM', correct4exc = True, correct4viscousDamping = False, forceFreq = True, e_bias = -0.085, e_adhoc_relative_error = 0.03, plotit = True, mock = {'on': False, 'k': 1e-2, 'Q': 50}):
    """
    Fit successive sines on data and residuals
    """
    s = lsq(_su, _is, _axis, mode, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error)
    s.showFFT()
    #s.fitAndCorrectEarthg(polynomial = True)
    if mock['on']:
        if not 'k' in mock:
            mock['k'] = 1e-2
        if not 'Q' in mock:
            mock['Q'] = 50
        s.addMockWire(mock['k'], mock['Q'])
    #kE = s.fit_eStiffness() #actually, contains *all* "pure" stiffness terms (electrostatic, gold wire if Q>>1, Newton, Yukawa)
    kE, gw_k, gw_Q, kQ, Delta_phase, lvdgw, res = s.fit_egwStiffness(forceFreq = forceFreq, correct4exc = correct4exc, correct4viscousDamping = correct4viscousDamping, plotit = plotit) #actually, kE contains *all* "pure" stiffness terms (electrostatic, gold wire if Q>>1, Newton, Yukawa)
    amp, freq, phase = s.fit_residualSine(plotit = plotit)

    th_kE = s.getTheoryElectrostaticStiffness()
    th_kN = s.getNewtonStiffness()

    print("\n##################################################")
    print("---> k_fits results")
    print("Overall (on-phase) stiffness (kE + gold wire if Q>>1 + Newton + Yukawa)")
    print("   kexc", correct4exc, s.omega_exc**2)
    print("   kE = " + str(1e3*kE) + " x1e-3 N/m")
    print("Gold wire")
    print("-- viscous damping (propto velocity, pi/2 out-of-phase)", correct4viscousDamping)
    print("     lambda_gw = " + str(lvdgw) + " N.s/m")
    print("-- internal damping (contained in kE if Q>>1, so should be closed to 0 in this case)")
    print("     GW stiffness = " + str(gw_k) + " N/m")
    print("     GW Q = " + str(gw_Q))
    print("     GW k/Q = " + str(kQ) + " N/m")
    print("     Phase difference [rad]", Delta_phase)
    print("Residual lingering sine")
    print("   amplitude: " + str(amp) + " m/s^2")
    print("   freq: " + str(freq) + " Hz")
    print("   phase: " + str(phase) + " rad")
    print()
    print("Theory")
    print("  Electrostatic = " + str(1e3*th_kE) + "  x1e-3 N/m")
    print("  Newtonian gravity = " + str(1e8*th_kN) + " x1e-8 N/m")
    print("     offset: measured - theory(kE) = " + str(kE - th_kE) + " N/m")
    print("     offset: measured - theory(kE+kN) = " + str(1e3*(kE - th_kE - th_kN)) + " x1e-3 N/m")
    print("##################################################")

    params = {'kexc': s.omega_exc**2, 'kE': kE, 'lambda_gw': lvdgw, 'gw_k': gw_k, 'gw_Q': gw_Q, 'k/Q': kQ, 'phi': Delta_phase, 'kN': th_kN, 'th_kE': th_kE, 'Deltak': kE - th_kE - th_kN}
    
    delta2theory = kE - th_kE - th_kN
    return delta2theory, params

def getAlpha(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM', lmbda = None, e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 0, plotit = False, noshow = False):
    s = suk(_su, _is, _axis, mode, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error)
    #k_delta = k_fits(_su = _su, _is = _is, _axis = _axis, correct4exc = True, correct4viscousDamping = correct4viscousDamping, forceFreq = True, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, plotit = plotit)
    #kY_est = k_delta

    kxQ = kQratios['X-' + _su]
    kyQ = kQratios['Y-' + _su]
    kzQ = kQratios['Z-' + _su]
    radial = k_fit_radial(_su = _su, _is = _is, kyQ = kyQ, kzQ = kzQ, taylorOrder = 4, phase = 'position', e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error, plotit = False)
    Q = radial['Q']
    if _axis == 'Y':
        kY_est = radial['k_corr_y' + mode]
    elif _axis == 'Z':
        kY_est = radial['k_corr_z' + mode]
    elif _axis == 'X':
        _res = k_fit(_su = _su, _is = _is, _axis = _axis, mode = mode, fit_kw = False, fit_lambda = True, kQ = kxQ, Q = Q, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error, plotit = False)
        kY_est = _res['k_corr']
    else:
        raise ValueError('WTH!?!')


    print('\n--> getAlpha: Yukawa stiffness', kY_est, '\n')
    
    if lmbda is None:
        nl = 15
        lmbda = np.logspace(-4, 0, nl)
    else:
        nl = np.size(lmbda)
    alpha = []
    alpha_err = []
    for ll in lmbda:
        k_th_alpha = s.getYukawaStiffness(1, ll) #k/alpha
        alpha.append(kY_est.nominal_value / k_th_alpha)
        alpha_err.append(kY_est.std_dev / np.abs(k_th_alpha))
        #print("----> hahaha...", kY_est, alpha, alpha_err, kY_est / k_th_alpha)
        #alpha.append((kY_est / k_th_alpha).nominal_value)
        #alpha_err.append((kY_est / k_th_alpha).std_dev)
    alpha = np.array(alpha)
    alpha_err = np.array(alpha_err)

    # plt.errorbar(lmbda, alpha, yerr = alpha_err, linestyle = '', marker = 'd')
    # plt.xlabel(r'$\lambda [m]$')
    # plt.ylabel(r'$\alpha$')
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.show(block = True)

    # plt.errorbar(lmbda, np.abs(alpha), yerr = alpha_err, linestyle = '', marker = 'd')
    # plt.xlabel(r'$\lambda [m]$')
    # plt.ylabel(r'$|\alpha|$')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(1e-4, 1e10) #as in Adelberger 2003
    # plt.tight_layout()
    # plt.show(block = True)

    #limits bounds at 2sigma
    upper_bounds_lin = alpha + 2 * alpha_err
    lower_bounds_lin = alpha - 2 * alpha_err

    # plt.plot(lmbda, alpha, color = 'black')
    # plt.fill_between(lmbda, lower_bounds_lin, upper_bounds_lin, color = 'grey')
    # plt.xlabel(r'$\lambda [m]$')
    # plt.ylabel(r'$\alpha$')
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.show(block = True)

    upper_bounds_log = []
    lower_bounds_log = []
    for i in range(nl):
        if np.sign(upper_bounds_lin[i] * lower_bounds_lin[i]) == 1:
            if alpha[i] > 0:
                upper_bounds_log.append(upper_bounds_lin[i])
                lower_bounds_log.append(lower_bounds_lin[i])
            else:
                upper_bounds_log.append(np.abs(lower_bounds_lin[i]))
                lower_bounds_log.append(np.abs(upper_bounds_lin[i]))
        else:
            if alpha[i] > 0:
                upper_bounds_log.append(np.abs(upper_bounds_lin[i]))
                lower_bounds_log.append(0)
            else:
                upper_bounds_log.append(np.abs(lower_bounds_lin[i]))
                lower_bounds_log.append(0)

    if not noshow:
        plt.plot(lmbda, np.abs(alpha), color = 'black')
        plt.fill_between(lmbda, upper_bounds_log, lower_bounds_log, color = 'grey')
        plt.xlabel(r'$\lambda [m]$')
        plt.ylabel(r'$\alpha$')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-4, 1e10) #as in Adelberger 2003
        plt.suptitle(_su + _is + _axis)
        plt.tight_layout()
        plt.show(block = True)

    return lmbda, alpha, alpha_err, kY_est


def combineAlpha(mode = 'both', lmbda = None, e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 0, radialOnly = True, plotit = False, show_kY = False):
    if mode == 'both':
        modes = ['HRM', 'FRM']
    elif mode in ['HRM', 'FRM']:
        modes = [mode]
    else:
        raise ValueError('Bad mode')
    if radialOnly:
        _ax = ['Y', 'Z']
    else:
        _ax = ['X', 'Y', 'Z']
    alpha = None
    mi = 0
    wkY_HRM = 0
    wkY_err_HRM = 0
    wkY_FRM = 0
    wkY_err_FRM = 0
    for mod in modes:
        mi += 1
        n = 0
        for _su in ['SUEP', 'SUREF']:
            for _is in ['IS1', 'IS2']:
                for _axis in _ax:
                    n += 1
                    lmbda, alpha_i, alpha_err_i, kYi = getAlpha(_su = _su, _is = _is, _axis = _axis, mode = mod, lmbda = lmbda, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error, plotit = False, noshow = (not plotit))
                    if alpha is None:
                        alpha = alpha_i
                        alpha_err = alpha_err_i
                    else:
                        alpha = np.vstack((alpha, alpha_i))
                        alpha_err = np.vstack((alpha_err, alpha_err_i))
                    if n == 1:
                        kY = [kYi.nominal_value]
                        kY_err = [kYi.std_dev]
                        if mi == 1:
                            label = [_su + '-' + _is + '-' + _axis]
                    else:
                        kY.append(kYi.nominal_value)
                        kY_err.append(kYi.std_dev)
                        if mi == 1:
                            label.append(_su + '-' + _is + '-' + _axis)

        #plot kY (iteratively, to have both modes for a same sensor/axis at the same x-position)
        if show_kY:
            kY = np.array(kY)
            kY_err = np.array(kY_err)
            x = np.linspace(1, n, n)
            if mod == 'HRM':
                marker = 'd'
            else:
                marker = 's'
            plt.errorbar(x, kY, yerr = kY_err, marker = marker, linestyle = '', color = 'black')
            if mod == 'HRM':
                wkY_HRM, wkY_err_HRM = weightedAverage(kY, kY_err)
            else:
                wkY_FRM, wkY_err_FRM = weightedAverage(kY, kY_err)
                
    if show_kY:
        print('kYs')
        print('   HRM', wkY_HRM, wkY_err_HRM)
        print('   FRM', wkY_FRM, wkY_err_FRM)
        if mode == 'HRM' and wkY_err_HRM > 0:
            plt.plot([0, n+1], [wkY_HRM, wkY_HRM], color = 'black', linestyle = '--')
            plt.fill_between([0, n+1], [wkY_HRM - wkY_err_HRM, wkY_HRM - wkY_err_HRM], [wkY_HRM + wkY_err_HRM, wkY_HRM + wkY_err_HRM], color = 'grey', alpha = 0.7)
        plt.plot([0, n+1], [0,0], color = 'black')
        plt.xlim(0, n+1)
        plt.xticks(x, tuple(label), rotation = 45)
        #plt.ylabel(r'$k_Y$ [N/m]')
        plt.ylabel(r'$\Delta k$ [N/m]')
        plt.tight_layout()
        plt.show(block = True)

    na = np.shape(alpha)[0]
    nl = np.shape(alpha)[1]
    wa = np.zeros(nl)
    wa_err = np.zeros(nl)
    for i in range(nl):
        wa[i], wa_err[i] = weightedAverage(alpha[:,i], alpha_err[:,i])
        #for j in range(na):
        #    plt.plot(lmbda[i], alpha[j,i], color = 'black', marker = 'x', linestyle = '')
    #plt.xlabel(r'$\lambda$ [m]')
    #plt.ylabel('alpha')
    #plt.show()

    print(wa)
    print(wa_err)

    #limits bounds at 2sigma
    upper_bounds_lin = wa + 2 * wa_err
    lower_bounds_lin = wa - 2 * wa_err
    upper_bounds_log = []
    lower_bounds_log = []
    for i in range(nl):
        if np.sign(upper_bounds_lin[i] * lower_bounds_lin[i]) == 1:
            if wa[i] > 0:
                upper_bounds_log.append(upper_bounds_lin[i])
                lower_bounds_log.append(lower_bounds_lin[i])
            else:
                upper_bounds_log.append(np.abs(lower_bounds_lin[i]))
                lower_bounds_log.append(np.abs(upper_bounds_lin[i]))
        else:
            if wa[i] > 0:
                upper_bounds_log.append(np.abs(upper_bounds_lin[i]))
                lower_bounds_log.append(0)
            else:
                upper_bounds_log.append(np.abs(lower_bounds_lin[i]))
                lower_bounds_log.append(0)

    plt.plot(lmbda, np.abs(wa), color = 'black')
    plt.fill_between(lmbda, upper_bounds_log, lower_bounds_log, color = 'grey')
    plt.xlabel(r'$\lambda [m]$')
    plt.ylabel(r'$\alpha$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(lmbda[0], lmbda[-1])
    plt.ylim(1e-4, 1e10) #as in Adelberger 2003
    plt.tight_layout()
    plt.show(block = True)

    if np.sum(lower_bounds_log) == 0:
        plt.fill_between(lmbda, upper_bounds_log, np.tile(1e10, nl), color = 'grey')
        plt.xlabel(r'$\lambda [m]$')
        plt.ylabel(r'$\alpha$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(lmbda[0], lmbda[-1])
        plt.ylim(1e-4, 1e10) #as in Adelberger 2003
        plt.tight_layout()
        plt.show(block = True)

    #save upper bounds in ascii file that can be read with yLimits.py under python2.7 to plot limits together with published ones
    with open('stiffnessYukawa.dat', 'w') as f:
        for i in range(np.size(lmbda)):
            f.writelines(str(lmbda[i]) + '\t' + str(upper_bounds_log[i]) + '\n')
        f.close()
    

def weightedAverage(val, err):
    """inverse variance weight"""
    w = 1./err**2
    num = np.sum(w * val)
    denom = np.sum(w)
    m = num / denom
    werr = 1./np.sqrt(np.sum(w))
    return m, werr
    
    
######################################################
def tst(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM', modelType = 'stiffness', nsteps = 20000):
    s = mcmc(_su, _is, _axis, mode, modelType = modelType, correct4theory_kE = False, plotData = False)
    #s.getErrors(plotit = False)
    #s.getExcitationCharacteristics(fitFrequency = False, plotit = False)
    #s.fitStiffness()
    s.maxL(init_error = 0.1, printit = True, plotit = True)
    s.mcmc(init_error = 0.1, nsteps = nsteps)
    s.plotWalkers()
    s.plotPosterior()

def tstLSQMatrix():
    t = np.arange(0, 2000, 0.25)
    omega = 2 * np.pi / 300
    psi = 0
    A = lsqMatrixModel(t, omega, psi, plotit = True)
    print(np.shape(A))
    
def tstLSQ(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM'):
    s = lsq(_su, _is, _axis, mode, correct4theory_kE = False, plotData = False)
    s.fitModel(recursive = True, plotit = True)


