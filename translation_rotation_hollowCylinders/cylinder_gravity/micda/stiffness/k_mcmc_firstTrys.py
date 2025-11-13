import numpy as np
import matplotlib.pyplot as plt
from micda.stiffness import rdStiffnessData
from pytsa.analysis.paramsEstimation import fitSingleSine
from uncertainties import ufloat
from scipy.optimize import minimize
import emcee
import corner

#theoretical electrostatic stiffness [N/m]
kth = {'SUEP_IS1_Y': -2.02e-2,
           'SUEP_IS1_Z': -2.02e-2,
           'SUEP_IS2_Y': -9.1e-2,
           'SUEP_IS2_Z': -9.1e-2,
           'SUREF_IS1_Y': -2.02e-2,
           'SUREF_IS1_Z': -2.02e-2,
           'SUREF_IS2_Y': -9.55e-2,
           'SUREF_IS2_Z': -9.55e-2}

#masses [kg]
mass = {'SUEP_IS1': 0.401706,
            'SUEP_IS2': 0.300939,
            'SUREF_IS1': 0.401533,
            'SUREF_IS2': 1.359813}


#starting values
k_elec_th_0 = 0.
k_cham_0 = 0.
b_0 = 0.

#allowed ranges
k_elec_th_range = [-10e-2, 10e-2]
k_cham_range = [-5e-2, 5e-2]
b_range = [-1e-6, 1e-6]

    
def fun(x, k_elec_th_m, k_cham_m, b, kth_m = 0):
    """
    Compute acceleration for a given position
    k_elec_th_m -- (k_elec - k_th)/m, m=mass
    k_cham_m -- k_cham/m
    b -- bias
    """
    Gamma = k_elec_th_m * x + k_cham_m * x + b + kth_m * x
    return Gamma

def funK(x, k_m, b, kth_m = 0):
    """
    Compute acceleration for a given position
    k_elec_th_m -- (k_elec - k_th)/m, m=mass
    k_cham_m -- k_cham/m
    b -- bias
    """
    Gamma = k_m * x + b + kth_m * x
    return Gamma

def log_likelihood(theta, x, y, yerr, kth_m = 0):
    k_elec_th_m, k_cham_m, b = theta
    model = fun(x, k_elec_th_m, k_cham_m, b, kth_m = kth_m)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

def log_likelihoodK(theta, x, y, yerr, kth_m = 0):
    k_m, b = theta
    model = funK(x, k_m, b, kth_m = kth_m)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))


class suk:
    def __init__(self, _su, _is, _axis, estimateTotalK = False, ignore_kth = False, plotData = False):
        #if estimateTotalK:
        #    ignore_kth = True
        self.estimateTotalK = estimateTotalK
        self._su = _su
        self._is = _is
        self._axis = _axis
        self.t, self.acc, self.pos = rdStiffnessData.getData(_su, _is, _axis, plotit = plotData)
        self.m = mass[_su + '_' + _is]
        self.kth = kth[_su + '_' + _is + '_' + _axis]
        self.kth_m = self.kth / self.m
        if ignore_kth:
            self.kth = 0
            self.kth_m = 0
        if not estimateTotalK:
            self.nparams = 3 #params to estimate: k_elec-k_th, k_cham, bias
        else:
            self.nparams = 2 #k, bias
        self.nwalkers = 32
        self.getErrors()

    def getErrors(self, plotit = False):
        """
        Get measurement errors on acceleration (resampled at 1 Hz). Just fit a sine and take residuals as errors.
        """
        freq = 2*np.pi / 300 #Hz
        fit = fitSingleSine(self.t, self.acc, freq_range = [1./500,1./200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit)
        self.acc_errors = fit['residuals']
        #self.acc_errors = getErrors(self.t, self.acc, plotit = plotit)

    def getExcitationCharacteristics(self, fitFrequency = False, plotit = False):
        """
        Get characteristics of position excitation: amplitude, frequency and phase
        """
        if not fitFrequency:
            freq = 2*np.pi / 300 #Hz
        else:
            freq = None
        fit = fitSingleSine(self.t, self.pos, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit)
        self.x0 = ufloat(fit['amplitude'], fit['err_amplitude'])
        self.xmean = ufloat(fit['mean'], fit['err_mean'])
        self.xphase = ufloat(fit['phase'], fit['err_phase'])
        self.xfreq = ufloat(fit['frequency'], 0) / (2.*np.pi)
        #print(fit)
        #print(self.x0)
        #print(self.xmean)
        #print(self.xphase)
        #print(self.xfreq)

    def getAccelerationSine(self, fitFrequency = False, plotit = False):
        if not fitFrequency:
            freq = 2*np.pi / 300 #Hz
        else:
            freq = None
        fit = fitSingleSine(self.t, self.acc, freq_range = [2*np.pi/500,2*np.pi/200], nfreq = 500, freq = freq, phase = None, rms = None, log = False, fit_drift = False, chisq_tol = 1e-3, lackConvergence = True, iterNoiseEstimation = True, nmaxIterNoise = 13, noiseEstimate_tol = 0.1, nFreqRefine = 0, freqRefineTolerance = 1e-7, plotit = plotit)
        self.a0 = ufloat(fit['amplitude'], fit['err_amplitude'])
        self.amean = ufloat(fit['mean'], fit['err_mean'])
        self.aphase = ufloat(fit['phase'], fit['err_phase'])
        self.afreq = ufloat(fit['frequency'], 0) / (2.*np.pi)

    def fitStiffness(self):
        """
        Compute naive stiffness as ratio between acceleration and position maxima
        """
        self.getExcitationCharacteristics()
        self.getAccelerationSine()
        k = -(self.a0 / self.x0) * self.m
        print(k)
        print(k*100)

        
    def initMaxL(self, init_error = 0.):
        k_elec_th_m_0 = k_elec_th_0 / self.m
        k_cham_m_0 = k_cham_0 / self.m
        if not self.estimateTotalK:
            zero = np.array([k_elec_th_m_0, k_cham_m_0, b_0])
        else:
            zero = np.array([k_elec_th_m_0 + k_cham_m_0, b_0])
        initial = zero * (1 + init_error * np.random.rand(self.nparams))
        return initial
        
    def maxL(self, init_error = 0., printit = False, plotit = False):
        if not self.estimateTotalK:
            nll = lambda *args: -log_likelihood(*args)
        else:
            nll = lambda *args: -log_likelihoodK(*args)
        initial = self.initMaxL(init_error = init_error)
        x = self.pos
        y = self.acc
        yerr = self.acc_errors
        soln = minimize(nll, initial, args = (x, y, yerr, self.kth_m))
        if not self.estimateTotalK:
            k_elec_th_m_ml, k_cham_m_ml, b_ml = soln.x
        else:
            k_m_ml, b_ml = soln.x

        if printit:
            print()
            print('ML estimates')
            if not self.estimateTotalK:
                print('k_elec-k_th = ' + str(k_elec_th_m_ml * self.m))
                print('k_cham = ' + str(k_cham_m_ml * self.m))
                print('Total k = ' + str(k_elec_th_m_ml * self.m + k_cham_m_ml * self.m))
            else:
                print('Total k = ' + str(k_m_ml * self.m))
            print('b = ' + str(b_ml))
            print()

        if plotit:
            if not self.estimateTotalK:
                ml = fun(x, k_elec_th_m_ml, k_cham_m_ml, b_ml, kth_m = self.kth_m)
            else:
                ml = funK(x, k_m_ml, b_ml, kth_m = self.kth_m)
            plt.errorbar(x, y, yerr = yerr, marker = '.', linestyle = '', color = 'black', label = 'Data')
            plt.plot(x, ml, color = 'blue', label = 'ML')
            plt.legend(loc = 'upper right')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.show(block = True)

        if not self.estimateTotalK:
            return k_elec_th_m_ml, k_cham_m_ml, b_ml, initial, soln
        else:
            return k_m_ml, b_ml, initial, soln

    
    def log_prior(self, theta):
        if not self.estimateTotalK:
            k_elec_th_m, k_cham_m, b = theta
        else:
            k_m, b = theta
        k_elec_th_m_min = k_elec_th_range[0] / self.m
        k_elec_th_m_max = k_elec_th_range[1] / self.m
        k_cham_m_min = k_cham_range[0] / self.m
        k_cham_m_max = k_cham_range[1] / self.m
        b_min = b_range[0]
        b_max = b_range[1]
        if not self.estimateTotalK:
            if k_elec_th_m_min < k_elec_th_m < k_elec_th_m_max and k_cham_m_min < k_cham_m < k_cham_m_max and b_min < b < b_max:
                return 0.
        else:
            if k_elec_th_m_min < k_m < k_elec_th_m_max and b_min < b < b_max:
                return 0.
        return -np.inf
        
    def log_probability(self, theta, x, y, yerr, kth_m = 0):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        if not self.estimateTotalK:
            return lp + log_likelihood(theta, x, y, yerr, kth_m = kth_m)
        else:
            return lp + log_likelihoodK(theta, x, y, yerr, kth_m = kth_m)

    def mcmc(self, init_error = 0., nsteps = 5000):
        x = self.pos
        y = self.acc
        yerr = self.acc_errors
        if not self.estimateTotalK:
            k_elec_th_m_ml, k_cham_m_ml, b_ml, initial, soln = self.maxL(init_error = init_error)
        else:
            k_m_ml, b_ml, initial, soln = self.maxL(init_error = init_error)
        pos = soln.x * (1 + init_error * np.random.rand(self.nwalkers, self.nparams))
        nwalkers, ndim = pos.shape
        #print("\nCochonne MCMC!")
        #print(k_elec_th_m_ml, k_cham_m_ml, b_ml)
        #print(self.log_prior(soln.x))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args = (x, y, yerr, self.kth_m))
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
        if not self.estimateTotalK:
            labels = ["(k_elec-k_th)/m", "k_cham/m", "b"]
        else:
            labels = ["k/m", "b"]
        for i in range(self.nparams):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        plt.show(block = True)

    def plotPosterior(self):
        if not self.estimateTotalK:
            labels = ["(k_elec-k_th)", "k_cham", "b"]
        else:
            labels = ["k", "b"]
        flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)
        print(flat_samples.shape)

        flat_samples[:,0] *= self.m
        flat_samples[:,1] *= self.m
        
        for i in range(self.nparams):
            res = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(res)
            print(labels[i], res[1], q[0], q[1])
    
        fig = corner.corner(flat_samples, labels=labels)
        plt.show(block = True)
    

def tst():
    s = suk('SUEP', 'IS2', 'Y', estimateTotalK = True, plotData = False, ignore_kth = True)
    #s.getErrors(plotit = False)
    #s.getExcitationCharacteristics(fitFrequency = False, plotit = False)
    s.fitStiffness()
    #k_elec_th_m_ml, k_cham_m_ml, b_ml, initial, soln = s.maxL(init_error = 1e-3, printit = True, plotit = True)
    s.maxL(init_error = 1e-3, printit = True, plotit = True)
    s.mcmc(init_error = 1e-3)
    s.plotWalkers()
    s.plotPosterior()
    
