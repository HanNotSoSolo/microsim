import numpy as np
import matplotlib.pyplot as plt
from  micda.daio import rdData
from pytsa.general.tsa_utils import parse_powerlaw_coeffs
from pytsa.analysis.tsa_stats import power_spectrum
import pytsa.analysis.spectral_analysis as spectral_analysis
import os.path
import micda.stiffness.temperatures as temp
from micda.stiffness.dataPath import getPath
from micda.stiffness import instrument
from uncertainties import ufloat, umath
from scipy.constants import k as kB


##masses [kg]
#mass = {'SUEP_IS1': 0.401706,
#            'SUEP_IS2': 0.300939,
#            'SUREF_IS1': 0.401533,
#            'SUREF_IS2': 1.359813}



def estimateAllKQs(session = 218, _su = 'SUEP', _is = 'IS1', Q = ufloat(100, 50), plotit = True):
    """
    Main wrap-up function. Measure k/Q for all axes of _su/_is for session, and k given prior Q.
    """
    noiseModel_fsep = None
    slopesCoeffsGuess = "{'-1': 1.3e-25}"
    slopesCoeffsGuess_2 = None
    noiseModelFreqRange = [5e-5,5e-3]
    k, kx, ky, kz, k_Q, kx_Q, ky_Q, kz_Q, phi, theta = estimate_kModulus(session, _su, _is, Q = Q, missingDataMngt = 'global mean', freq_range = noiseModelFreqRange, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = noiseModel_fsep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = 20, plotit = plotit)
    
    print("\n---> estimateAllKQs", session, _su, _is)
    print("   prior Q: " + str(Q.nominal_value) + "+/-" + str(Q.std_dev))
    print("   k/Q: " + str(k_Q.nominal_value) + "+/-" + str(k_Q.std_dev))
    print("   kx/Q: " + str(kx_Q.nominal_value) + "+/-" + str(kx_Q.std_dev))
    print("   ky/Q: " + str(ky_Q.nominal_value) + "+/-" + str(ky_Q.std_dev))
    print("   kz/Q: " + str(kz_Q.nominal_value) + "+/-" + str(kz_Q.std_dev))
    print("   k: " + str(k.nominal_value) + "+/-" + str(k.std_dev))
    print("   kx: " + str(kx.nominal_value) + "+/-" + str(kx.std_dev))
    print("   ky: " + str(ky.nominal_value) + "+/-" + str(ky.std_dev))
    print("   kz: " + str(kz.nominal_value) + "+/-" + str(kz.std_dev))
    print("   phi", phi * 180 / np.pi)
    print("   theta", theta * 180 / np.pi)
    return k_Q, kx_Q, ky_Q, kz_Q, k, kx, ky, kz, phi, theta


def combineSessions(sessions, _su = 'SUEP', _is = 'IS1', Q = ufloat(100, 50), plotit = True):
    n = np.size(sessions)
    if n == 1:
        return estimateAllKQs(session = sessions, _su = _su, _is = _is, Q = Q, plotit = plotit)
    k_Q = ufloat(0,0)
    kx_Q = ufloat(0,0)
    ky_Q = ufloat(0,0)
    kz_Q = ufloat(0,0)
    k = ufloat(0,0)
    kx = ufloat(0,0)
    ky = ufloat(0,0)
    kz = ufloat(0,0)
    phi = ufloat(0,0)
    theta = ufloat(0,0)
    for i in range(n):
        k_Qi, kx_Qi, ky_Qi, kz_Qi, ki, kxi, kyi, kzi, phii, thetai = estimateAllKQs(session = sessions[i], _su = _su, _is = _is, Q = Q, plotit = plotit)
        k_Q += k_Qi
        kx_Q += kx_Qi
        ky_Q += ky_Qi
        kz_Q += kz_Qi
        k += ki
        kx += kxi
        ky += kyi
        kz += kzi
        phi += phii
        theta += thetai 

    k_Q /= n
    kx_Q /= n
    ky_Q /= n
    kz_Q /= n
    k /= n
    kx /= n
    ky /= n
    kz /= n
    phi /= n
    theta /= n
    
    print("\n---> combineSessions", sessions, _su, _is)
    print("   prior Q: " + str(Q.nominal_value) + "+/-" + str(Q.std_dev))
    print("   k/Q: " + str(k_Q.nominal_value) + "+/-" + str(k_Q.std_dev))
    print("   kx/Q: " + str(kx_Q.nominal_value) + "+/-" + str(kx_Q.std_dev))
    print("   ky/Q: " + str(ky_Q.nominal_value) + "+/-" + str(ky_Q.std_dev))
    print("   kz/Q: " + str(kz_Q.nominal_value) + "+/-" + str(kz_Q.std_dev))
    print("   k: " + str(k.nominal_value) + "+/-" + str(k.std_dev))
    print("   kx: " + str(kx.nominal_value) + "+/-" + str(kx.std_dev))
    print("   ky: " + str(ky.nominal_value) + "+/-" + str(ky.std_dev))
    print("   kz: " + str(kz.nominal_value) + "+/-" + str(kz.std_dev))
    print("   phi", phi * 180 / np.pi)
    print("   theta", theta * 180 / np.pi)
    return k_Q, kx_Q, ky_Q, kz_Q, k, kx, ky, kz, phi, theta
    
def estimate_k(session, _su, _is, _axis, Q = ufloat(100, 50), missingDataMngt = 'global mean', freq_range = None, slopesCoeffsGuess = None, freq_sep = None, slopesCoeffsGuess_2 = None, noise_smooth_width = 20, plotit = True):
    """
    Q -- wire quality factor (default from Willemenot & Touboul 2000)
    """
    k_Q = estimate_kQ(session, _su, _is, _axis, missingDataMngt = missingDataMngt, freq_range = freq_range, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = freq_sep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = noise_smooth_width, plotit = plotit)
    k = k_Q * Q
    return k, k_Q


def estimate_kModulus(session, _su, _is, Q = ufloat(100, 50), missingDataMngt = 'global mean', freq_range = None, slopesCoeffsGuess = None, freq_sep = None, slopesCoeffsGuess_2 = None, noise_smooth_width = 20, plotit = True):
    kx, kx_Q = estimate_k(session, _su, _is, 'X', Q = Q, missingDataMngt = missingDataMngt, freq_range = freq_range, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = freq_sep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = noise_smooth_width, plotit = plotit)
    ky, ky_Q = estimate_k(session, _su, _is, 'Y', Q = Q, missingDataMngt = missingDataMngt, freq_range = freq_range, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = freq_sep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = noise_smooth_width, plotit = plotit)
    kz, kz_Q = estimate_k(session, _su, _is, 'Z', Q = Q, missingDataMngt = missingDataMngt, freq_range = freq_range, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = freq_sep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = noise_smooth_width, plotit = plotit)

    k = umath.sqrt(kx**2 + ky**2 + kz**2)
    k_Q = umath.sqrt(kx_Q**2 + ky_Q**2 + kz_Q**2)
    kx_k = kx / k
    theta0 = np.arctan2(kz.nominal_value, ky.nominal_value)

    kxk = np.random.randn(1000) * kx_k.std_dev + kx_k.nominal_value
    phi_samples = np.arccos(kxk)
    phi0 = np.mean(phi_samples)
    phi_err = np.std(phi_samples, ddof = 1)
    phi = ufloat(phi0, phi_err)

    kzs = np.random.randn(1000) * kz.std_dev + kz.nominal_value
    kys = np.random.randn(1000) * ky.std_dev + ky.nominal_value
    theta_samples = np.arctan2(kzs, kys)
    theta0 = np.mean(theta_samples)
    theta_err = np.std(theta_samples, ddof = 1)
    theta = ufloat(theta0, theta_err)
        
    return k, kx, ky, kz, k_Q, kx_Q, ky_Q, kz_Q, phi, theta
    

def estimate_kQ(session, _su, _is, _axis, missingDataMngt = 'global mean', freq_range = None, slopesCoeffsGuess = None, freq_sep = None, slopesCoeffsGuess_2 = None, noise_smooth_width = 20, plotit = True):
    #fit low frequency acceleration PSD. Always consider IS1, since IS2 is the drag-free accelerometer, thus have no f^-1 spectrum
    t, acc4Hz, mask = getData(session, _su, 'IS1', _axis, plotit = plotit)
    bd = np.where(mask == 0)[0]
    nbd = np.size(bd)
    if nbd > 0:
        if missingDataMngt == 'ignore':
            pass
        elif missingDataMngt == 'trim':
            if nbd > 20:
                print('WARNING! Quite a lot of missing data (' + str(nbd) + '). Triming them may be dangerous')
            t = t[mask == 1]
            acc4Hz = acc4Hz[mask == 1]
        elif missingDataMngt == 'global mean':
            m = np.mean(acc4Hz[mask == 1])
            acc4Hz[bd] = m
        else:
            raise NotImplementedError()
        
    coeff, error = fitLowFrequencySpectrum(t, acc4Hz, freq_range = freq_range, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = freq_sep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = noise_smooth_width, plotit = plotit)
    fac = ufloat(coeff, error) #this is 4.kB.T / (2 pi * mass^2) * k/Q
    print("--> f^-1 coeff (power spectrum)", fac)

    #get average temperature
    t, T1, T2, T3, T4, T5, T6, TDet1, TDet2 = temp.getData(session, _su, plotit = plotit)
    mean_T1 = np.mean(T1)
    mean_T2 = np.mean(T2)
    mean_T3 = np.mean(T3)
    mean_T4 = np.mean(T4)
    mean_T5 = np.mean(T5)
    mean_T6 = np.mean(T6)
    T = np.mean(np.array([mean_T1, mean_T2, mean_T3, mean_T4, mean_T5, mean_T6]))
    print("--> mean temperature", T)
    
    #get mass of the test mass
    m = instrument.mass[_su + '_' + _is]
    print("--> mass", m)

    prefac = 4 * kB * T / m**2 / (2 * np.pi)
    k_Q = fac / prefac
    print("--> k/Q", k_Q)
    return k_Q #k/Q, k = wire stiffness, Q = wire quality factor, unit = N/m
    
    

def getData(session, _su, _is, _axis, plotit = False):
    if not _is in ['IS1', 'IS2']:
        raise ValueError("Bad IS", _is)

    if not _axis in ['X', 'Y', 'Z']:
        raise ValueError("Bad axis", _axis)

    if not _su in ['SUEP', 'SUREF']:
        raise ValueError("Bad SU", _su)
    
    #if session == 218:
    #    _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/Phase_3/Session_218_EPR_V3DFIS2_01_SUEP/N0c_01/"
    #else:
    #    raise NotImplementedError()
    _path = getPath(session, _su)

    if _axis in ['Y', 'Z']:
        acc_file = os.path.join(_path, _su, _is, 'Acceleration' + _axis + '.bin')
    else:
        acc_file = os.path.join(_path, _su, _is, 'AccelerationXscaa.bin')
    t, acc4Hz, mask = rdData(acc_file)
    t -= t[0]

    if session == 294:
        gd = np.where(t > 113000)[0] #b/c first 18 orbits are not good
        t = t[gd]
        acc4Hz = acc4Hz[gd]
        mask = mask[gd]
    elif session == 380:
        gd = np.where((t > 142000) & (t < 640000))[0] #first and last few orbits are not good
        t = t[gd]
        acc4Hz = acc4Hz[gd]
        mask = mask[gd]
    else:
        pass
    
    if plotit:
        ax = plt.subplot(211)
        plt.plot(t, acc4Hz)
        plt.ylabel(r'acc [ms$^{-2}$]')
        plt.subplot(212, sharex = ax)
        plt.plot(t, mask)
        plt.ylim(-0.1, 1.1)
        plt.ylabel('Mask')
        plt.xlabel('t [s]')
        plt.show()
        
    return t, acc4Hz, mask


def fitLowFrequencySpectrum(t, data, freq_range = None, slopesCoeffsGuess = None, freq_sep = None, slopesCoeffsGuess_2 = None, noise_smooth_width = 20, plotit = True):
    #NB: need onesided = True to be consistent with simTimSeries
    freqs, powspec, enbw = power_spectrum(np.max(t), data, tmin = np.min(t), fft_out = False, hann = True, correct4mean = True, power_of_two = False, onesided = True)
    noise_psd = powspec / enbw

    #fit noise with power law
    f = freqs[freqs > 0]
    smooth_width = noise_smooth_width
    aps_noise = spectral_analysis.spectrum(freqs, np.sqrt(powspec), enbw = enbw, smoothprop = {'signal_method': 'daniell', 'continuum_method': 'daniell', 'continuum_width': 200, 'signal_width': smooth_width})
    if smooth_width > 0:
        aps_noise.smooth() #smooth if required
    slopes1, coeffs1, nc1 = parse_powerlaw_coeffs(slopesCoeffsGuess) #guess parameters for the fit
    slopes1 = np.array(slopes1)
    p0 = coeffs1
    if freq_range is None:
        freq_ran = [np.min(f), np.max(f)]
    else:
        freq_ran = [max(np.min(f), freq_range[0]), min(np.max(f), freq_range[1])]
    if slopesCoeffsGuess_2 is None:
        null_params1, null_cov1 = aps_noise.cmp_null_dynamic(slopes1, p0 = p0, freq_range = freq_ran, returnParams = True) #null_params and null_cov are for PSD
        slopes = slopes1
        cs = null_params1
        errors = np.sqrt(np.diag(null_cov1))
    else:
        if freq_sep is None:
            raise TypeError("freq_sep must be set to perform noise fit on two segments!")
        slopes2, coeffs2, nc2 = parse_powerlaw_coeffs(slopesCoeffsGuess_2) #guess parameters for the fit
        slopes2 = np.array(slopes2)
        p02 = coeffs2
        null_params1, null_cov1, null_params2, null_cov2 = aps_noise.cmp_null_dynamic(slopes1, slopes2 = slopes2, p0 = p0, p02 = p02, freq_range = freq_ran, fsep = freq_sep, returnParams = True)
        slopes = np.append(slopes1, slopes2)
        cs = np.append(null_params1, null_params2)
        errors = np.append(np.sqrt(np.diag(null_cov1)), np.sqrt(np.diag(null_cov2)))
    #print("fitLowFrequencySpectrum", slopes, cs, errors)

    if plotit:
        model2 = 0
        for i in range(np.size(slopes)):
            gd = np.where((f >= freq_ran[0]) & (f <= freq_ran[1]))
            fm = f[gd]
            model2 += cs[i] * fm**slopes[i]
        nmc = 100
        mcModels = monteCarloModels(nmc, fm, slopes, cs, errors)
        psd = noise_psd[freqs > 0]
        plt.loglog(f, np.sqrt(psd), color = 'black')
        #plt.loglog(f, np.sqrt(powspec[freqs > 0]), color = 'black')
        plt.loglog(fm, np.sqrt(model2), color = 'red')
        for i in range(nmc):
            plt.loglog(fm, np.sqrt(mcModels[i]), color = 'orange', alpha = 0.5)
        plt.xlabel('Freq [Hz]')
        plt.ylabel(r'Acceleration spectral density [ms$^{-2}/\sqrt{\rm Hz}$]')
        plt.show()

    gd = np.where(slopes == -1)[0]
    if np.size(gd) <= 0:
        raise ValueError("Hum, too bad! No -1 slope in the fit, though it is the one of the gold wire...")
    coeff = cs[gd] #* enbw #m^2 s^-4
    error = errors[gd] #* enbw #m^2 s^-4
    return coeff[0], error[0]


def monteCarloModels(n, f, slopes, cs, errors):
    """
    Monte Carlo drawings of modeled PSD (for plotting purposes only)
    """
    models = []
    for i in range(n):
        imodel = 0
        for j in range(np.size(slopes)):
            jcoeff = np.random.rand(1) * errors[j] + cs[j]
            imodel += jcoeff * f**slopes[j]
        models.append(imodel)
    return models

        


def tst(session = 218, _su = 'SUEP', _is = 'IS1', _axis = 'X'):
    #slopesCoeffsGuess = "{'-1': 1.3e-25, '4': 2.71e-20}"
    ##slopesCoeffsGuess_2 = "{'4': 2.e-20}"
    #slopesCoeffsGuess_2 = None
    #noiseModelFreqRange = [5e-5,0.5]
    noiseModel_fsep = 4e-2
    slopesCoeffsGuess = "{'-1': 1.3e-25}"
    slopesCoeffsGuess_2 = None
    noiseModelFreqRange = [5e-5,5e-3]
    k, k_Q = estimate_k(session, _su, _is, _axis, missingDataMngt = 'trim', freq_range = noiseModelFreqRange, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = noiseModel_fsep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = 20, plotit = True)

def tstModulus(session = 218, _su = 'SUEP', _is = 'IS1', plotit = True):
    noiseModel_fsep = None
    slopesCoeffsGuess = "{'-1': 1.3e-25}"
    slopesCoeffsGuess_2 = None
    noiseModelFreqRange = [5e-5,5e-3]
    k, kx, ky, kz, k_Q, kx_Q, ky_Q, kz_Q, phi, theta = estimate_kModulus(session, _su, _is, Q = ufloat(50, 45), missingDataMngt = 'global mean', freq_range = noiseModelFreqRange, slopesCoeffsGuess = slopesCoeffsGuess, freq_sep = noiseModel_fsep, slopesCoeffsGuess_2 = slopesCoeffsGuess_2, noise_smooth_width = 20, plotit = plotit)
    print("   k", k)
    print("   kx", kx)
    print("   ky", ky)
    print("   kz", kz)
    print("   k/Q", k_Q)
    print("   kx/Q", kx_Q)
    print("   ky/Q", ky_Q)
    print("   kz/Q", kz_Q)
    print("   phi", phi * 180 / np.pi)
    print("   theta", theta * 180 / np.pi)
