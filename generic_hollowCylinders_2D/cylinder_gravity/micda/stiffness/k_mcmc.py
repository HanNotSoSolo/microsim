raise NotImplementedError("Obsolete. Replaced by k_analysis.py")

import numpy as np
import matplotlib.pyplot as plt
from micda.stiffness import rdStiffnessData, instrument, eStiffness
from micda.stiffness.model import mkModel
import micda.stiffness.microscopeCylindersGravity as mcg
from pytsa.analysis.paramsEstimation import fitSingleSine
from uncertainties import ufloat
from scipy.optimize import minimize
import emcee
import corner


#starting values (kE and b set dynamically)
alpha_0 = 1
lmbda_0 = 0.01
#kE_0 = 'theory'
kGW_0 = 1e-3
QGW_0 = 1000
#b_0 = 0

#allowed ranges (bias and k_E based pre-fit and theory)
alpha_range = [-1e5, 1e5]
lmbda_range = [1e-4, 100]
kGW_range = [0, 0.1]
QGW_range = [1, 10000]


def fun(_is, _su, _axis, t, alpha, lmbda, kE, kGW, QGW, b, psi = 0, correct4theory_kE = False, plotit = False):
    #print("---> fun, ", alpha, lmbda, kE, kGW, QGW, b, psi)
    if lmbda < 0:
        raise ValueError('lmbda must be > 0')
    Gamma = mkModel(_is, _su, _axis, alpha, lmbda, t = t, x0 = 5e-6, fexc = 1./300, psi = psi, kE = kE, kGW = kGW, QGW = QGW, b = b, correct4theory_kE = correct4theory_kE, plotit = plotit)
    return Gamma

def log_likelihood(theta, x, y, yerr, _is = None, _su = None, _axis = None, psi = 0, correct4theory_kE = False):
    if _is is None or _su is None or _axis is None:
        raise TypeError("_is, _su and _axis must be set")
    alpha, lmbda, kE, kGW, QGW, b = theta
    model = fun(_is, _su, _axis, x, alpha, lmbda, kE, kGW, QGW, b, psi = psi, correct4theory_kE = correct4theory_kE)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))


class suk:
    def __init__(self, _su, _is, _axis, mode, correct4theory_kE = False, e_bias = -0.085, e_adhoc_relative_error = 0.03, plotData = False):
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
        self.t, self.acc, self.pos = rdStiffnessData.getData(_su, _is, _axis, mode, plotit = plotData)
        self.omega_exc = 2.* np.pi / 300 #period of 300 sec
        self.m = instrument.mass[_su + '_' + _is]
        self.nparams = 6 #alpha, lmbda, kE, kGW, QGW, b
        self.nwalkers = 32
        self.getErrors()
        self.getExcitationCharacteristics()
        self.getAccelerationSine()
        if self._axis == 'X':
            kE_0 = 0
            kerr1 = 0
            kerr2 = 0
            self.kE_0 = kE_0
            self.kE_rms0 = kerr1
        else:
            kE_0, kerr1, kerr2 = eStiffness.cmpStiffnessDistrib(_is = _is_int, _su = _su, mode = self.mode, errors = True, nsamples = 10000, bias = e_bias, adhoc_relative_error = e_adhoc_relative_error, plotDistrib = False)
            self.kE_0 = -kE_0 #careful to convention!!!
            self.kE_rms0 = kerr1 #could be used to define a range

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

        
        
    def getErrors(self, plotit = False):
        """
        Get measurement errors on acceleration (resampled at 1 Hz). Just fit twosuccessive sines and take residuals as errors.
        """
        freq = 2*np.pi / 300 #rad/s
        fit = fitSingleSine(self.t, self.acc, freq_range = [1./500,1./200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit, silent = True)
        fit = fitSingleSine(self.t, fit['residuals'], freq_range = [1./500,1./200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit, silent = True)
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
        fit = fitSingleSine(self.t, self.pos, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
        self.x0 = ufloat(fit['amplitude'], fit['err_amplitude'])
        self.xmean = ufloat(fit['mean'], fit['err_mean'])
        self.xphase = ufloat(fit['phase'], fit['err_phase'])
        self.xfreq = ufloat(fit['frequency'], 0) / (2.*np.pi)
        print("---> getExcitationCharacteristics")
        print('x0:', self.x0)
        #print(self.xmean)
        print('phase:', self.xphase)
        print('freq:', self.xfreq)
        print('red. chisq:', fit['rchi2'])
        if plotit:
            plt.plot(self.t, self.pos, label = 'Data')
            plt.plot(fit['x'], fit['fit'], label = 'Fit')
            plt.plot(fit['x'], fit['residuals'], label = 'Residuals')
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel('pos [m]')
            plt.show(block = True)

    def getAccelerationSine(self, fitFrequency = False, plotit = False, correct4exc = False):
        if correct4exc:
            acc = self.acc - self.omega_exc**2 * self.pos
        else:
            acc = self.acc
        if not fitFrequency:
            freq = 2*np.pi / 300 #Hz
        else:
            freq = None
        fit = fitSingleSine(self.t, acc, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit, silent = True)
        self.a0 = ufloat(fit['amplitude'], fit['err_amplitude'])
        self.b_0 = ufloat(fit['mean'], fit['err_mean'])
        self.aphase = ufloat(fit['phase'], fit['err_phase'])
        self.afreq = ufloat(fit['frequency'], 0) / (2.*np.pi)
        return acc

    def fit_eStiffness(self, correct4exc = True):
        """
        Compute naive stiffness as ratio between acceleration and position maxima
        """
        #self.getExcitationCharacteristics()
        acc = self.getAccelerationSine(correct4exc = correct4exc)
        k = (self.a0 / self.x0) * self.m
        if self._axis in ['Y', 'Z']:
            #the minus sign is here b/c we know that position and acceleration are out of phase, but only for radial axes
            k *= -1
        print('---> fit_eStiffness', k)
        #print(self.a0, self.x0)
        #print(self.aphase, self.xphase, self.aphase - self.xphase, np.pi - (self.aphase - self.xphase))
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
        resFit = fitSingleSine(self.t, res, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
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
        resFit = fitSingleSine(self.t, res, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = phase, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
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
        resFit = fitSingleSine(self.t, res, freq_range = [2*np.pi/1500,2*np.pi/20], nfreq = 500, freq = None, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = False, silent = True)
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
    
    def initMaxL(self, init_error = 0.):
        zero = np.array([alpha_0, lmbda_0, self.kE_0, kGW_0, QGW_0, self.b_0.nominal_value])
        initial = zero * (1 + init_error * np.random.rand(self.nparams))
        return initial
        
    def maxL(self, init_error = 0., printit = False, plotit = False):
        """
        log_likelihood() signature: log_likelihood(theta, x, y, yerr, _is = None, _su = None, _axis = None, correct4theory_kE = False)
        """
        nll = lambda *args: -log_likelihood(*args)
        initial = self.initMaxL(init_error = init_error)
        bnds = ((None, None), (0, None), (-2*np.abs(self.kE_0), 2*np.abs(self.kE_0)), (0, 0.1), (1, 100000), (self.b_0.nominal_value - 5 * self.b_0.std_dev, self.b_0.nominal_value + 5 * self.b_0.std_dev))
        #x = self.pos
        x = self.t
        y = self.acc
        yerr = self.acc_errors
        soln = minimize(nll, initial, args = (x, y, yerr, self._is, self._su, self._axis, self.xphase.nominal_value, self.correct4theory_kE), bounds = bnds)
        alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml = soln.x

        if printit:
            print()
            print('\n---> ML estimates')
            print('kE = ' + str(kE_ml))
            print('kGW = ' + str(kGW_ml))
            print('QGW = ' + str(QGW_ml))
            print('b = ' + str(a0_ml))
            print('alpha = ' + str(alpha_ml))
            print('lambda = ' + str(lmbda_ml))
            print()

        if plotit:
            ml = fun(self._is, self._su, self._axis, self.t, alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml, psi = self.xphase.nominal_value, correct4theory_kE = self.correct4theory_kE)
            plt.errorbar(x, y, yerr = yerr, marker = '.', linestyle = '', color = 'black', label = 'Data')
            plt.plot(x, ml, color = 'blue', label = 'ML')
            plt.legend(loc = 'upper right')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.show(block = True)

        return alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml, initial, soln

    
    def log_prior(self, theta):
        alpha, lmbda, kE, kGW, QGW, b = theta
        alpha_min = alpha_range[0]
        alpha_max = alpha_range[1]
        lmbda_min = lmbda_range[0]
        lmbda_max = lmbda_range[1]
        #kE_min = self.kE_0 - 5 * self.kE_rms0
        #kE_max = self.kE_0 + 5 * self.kE_rms0
        kE_min = -2 * np.abs(self.kE_0)
        kE_max = 2 * np.abs(self.kE_0)
        kGW_min = kGW_range[0]
        kGW_max = kGW_range[1]
        QGW_min = QGW_range[0]
        QGW_max = QGW_range[1]
        a0_min = self.b_0.nominal_value - 15 * self.b_0.std_dev
        a0_max = self.b_0.nominal_value + 15 * self.b_0.std_dev
        
        if alpha_min < alpha < alpha_max and lmbda_min < lmbda < lmbda_max and kE_min < kE < kE_max and kGW_min < kGW < kGW_max and QGW_min < QGW < QGW_max and a0_min < b < a0_max:
            return 0.
        return -np.inf
        
    def log_probability(self, theta, x, y, yerr):
        """
        log_likelihood signature: log_likelihood(theta, x, y, yerr, _is = None, _su = None, _axis = None, psi = 0, correct4theory_kE = False)
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr, _is = self._is, _su = self._su, _axis = self._axis, psi = self.xphase.nominal_value, correct4theory_kE = self.correct4theory_kE)


    def mcmc(self, init_error = 0., nsteps = 5000):
        x = self.t
        y = self.acc
        yerr = self.acc_errors
        alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml, initial, soln = self.maxL(init_error = init_error)
        pos = soln.x * (1 + init_error * np.random.rand(self.nwalkers, self.nparams))
        nwalkers, ndim = pos.shape
        #print("\nCochonne MCMC!")
        #print(alpha_ml, lmbda_ml, kE_ml, kGW_ml, QGW_ml, a0_ml)
        #print(self.log_prior(soln.x))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args = (x, y, yerr))
        sampler.run_mcmc(pos, nsteps, progress = True)

        tau = sampler.get_autocorr_time()
        print('MCMC autocorr time', tau)
        
        self.sampler = sampler

    
    def plotWalkers(self):
        """
        sampler -- mcmc output
        """
        fig, axes = plt.subplots(self.nparams, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["alpha", "lambda", "kE", "kGW", "Q", "b"]
        for i in range(self.nparams):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        plt.show(block = True)

    def plotPosterior(self):
        labels = ["alpha", "lambda", "kE", "kGW", "Q", "b"]
        flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)
        print(flat_samples.shape)

        for i in range(self.nparams):
            res = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(res)
            print(labels[i], res[1], q[0], q[1])
    
        fig = corner.corner(flat_samples, labels=labels)
        plt.show(block = True)
    



def k_fits(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM', correct4exc = True, correct4viscousDamping = False, forceFreq = True, e_bias = -0.085, e_adhoc_relative_error = 0.03, plotit = True, mock = {'on': False, 'k': 1e-2, 'Q': 50}):
    s = suk(_su, _is, _axis, mode, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error)
    #s.fitAndCorrectEarthg(polynomial = True)
    if mock['on']:
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

def getAlpha(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM', lmbda = None, correct4viscousDamping = False, e_bias = -0.085, e_adhoc_relative_error = 0.03, plotit = False):
    s = suk(_su, _is, _axis, mode, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error)
    k_delta = k_fits(_su = _su, _is = _is, _axis = _axis, correct4exc = True, correct4viscousDamping = correct4viscousDamping, forceFreq = True, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, plotit = plotit)
    kY_est = k_delta

    if lmbda is None:
        nl = 15
        lmbda = np.logspace(-4, 0, nl)
    else:
        nl = np.size(lmbda)
    alpha = []
    alpha_err = []
    for ll in lmbda:
        k_th_alpha = s.getYukawaStiffness(1, ll) #k/alpha
        alpha.append((kY_est / k_th_alpha).nominal_value)
        alpha_err.append((kY_est / k_th_alpha).std_dev)
    alpha = np.array(alpha)
    alpha_err = np.array(alpha_err)

    plt.errorbar(lmbda, alpha, yerr = alpha_err, linestyle = '', marker = 'd')
    plt.xlabel(r'$\lambda [m]$')
    plt.ylabel(r'$\alpha$')
    plt.xscale('log')
    plt.tight_layout()
    plt.show(block = True)

    plt.errorbar(lmbda, np.abs(alpha), yerr = alpha_err, linestyle = '', marker = 'd')
    plt.xlabel(r'$\lambda [m]$')
    plt.ylabel(r'$|\alpha|$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-4, 1e10) #as in Adelberger 2003
    plt.tight_layout()
    plt.show(block = True)

    #limits bounds at 2sigma
    upper_bounds_lin = alpha + 2 * alpha_err
    lower_bounds_lin = alpha - 2 * alpha_err

    plt.plot(lmbda, alpha, color = 'black')
    plt.fill_between(lmbda, lower_bounds_lin, upper_bounds_lin, color = 'grey')
    plt.xlabel(r'$\lambda [m]$')
    plt.ylabel(r'$\alpha$')
    plt.xscale('log')
    plt.tight_layout()
    plt.show(block = True)

    upper_bounds_log = []
    lower_bounds_log = []
    for i in range(nl):
        if np.sign(upper_bounds_lin[i] * lower_bounds_lin[i]) == 1:
            if alpha[i] > 0:
                upper_bounds_log.append(upper_bounds_lin[i])
                lower_bounds_log.append(lower_bounds_lin[i])
            else:
                upper_bounds_log.append(np.abs(upper_bounds_lin[i]))
                lower_bounds_log.append(np.abs(lower_bounds_lin[i]))
        else:
            if alpha[i] > 0:
                upper_bounds_log.append(np.abs(upper_bounds_lin[i]))
                lower_bounds_log.append(0)
            else:
                upper_bounds_log.append(np.abs(lower_bounds_lin[i]))
                lower_bounds_log.append(0)

    plt.plot(lmbda, np.abs(alpha), color = 'black')
    plt.fill_between(lmbda, upper_bounds_log, lower_bounds_log, color = 'grey')
    plt.xlabel(r'$\lambda [m]$')
    plt.ylabel(r'$\alpha$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-4, 1e10) #as in Adelberger 2003
    plt.tight_layout()
    plt.show(block = True)
    
    
######################################################
def tst(_su = 'SUEP', _is = 'IS1', _axis = 'Y'):
    s = suk(_su, _is, _axis, correct4theory_kE = False, plotData = False)
    #s.getErrors(plotit = False)
    #s.getExcitationCharacteristics(fitFrequency = False, plotit = False)
    s.fitStiffness()
    #s.maxL(init_error = 0.1, printit = True, plotit = True)
    s.mcmc(init_error = 0.1, nsteps = 25000)
    s.plotWalkers()
    s.plotPosterior()
