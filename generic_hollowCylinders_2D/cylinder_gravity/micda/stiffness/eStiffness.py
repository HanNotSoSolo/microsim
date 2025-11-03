import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0
try:
    from micda.stiffness.rdStiffnessData import getVpVd, getPositionVoltages
except:
    print("Could not import rdStiffnessData")
from pytsa.general.histOutline import histOutline

#see MIC-DC-S-7-TS-SU-7350-ONE-1.pdf for theoretical numerical values, and MIC-DC-S-7-TS-SU-7337-ONE-2.pdf for measured geometry of SUEP
#Rey = outer radius of Y and Z electrodes
#Rmi = inner radius of TM
#Ly = length of Y and Z electrodes
#d3 = width of dips between Y and Z electrodes
#Rp = outer radius of TM
#Rx = inner radius of X electrodes
#Rphi = inner radius of phi electrodes
#h = length of X electrodes
#Lphi = length of phi electrodes
is1_suep = {'Rey': 0.0148-1,
            'Rey_err': 2e-6,
           'Rmi': 0.0154005,
           'Rmi_err': 2e-6,
           'Ly': 0.015,
           'Ly_err': 2e-6,
           #'p1': 0.0015,
           #'p1_err': 1e-6,
           #'d1': 0.0015,
           #'d1_err': 1e-6,
           #'d4': 0.004,
           #'d4_err': 1e-6,
           'd3': 0.003,
           'd3_err': 2e-6,
           'Rp': 0.019695,
           'Rp_err': 2e-6,
           'Rx': 0.0203,
           'Rx_err': 2e-6,
           'Rphi': 0.0203,
           'Rphi_err': 2e-6,
           'h': 0.008165, #Lx in MIC-DC-S-7-TS-SU-7350-ONE-1.pdf
           'h_err': 2e-6,
           'Lphi': 0.019,
            'Lphi_err': 2e-6}

is2_suep = {'Rey': 0.0298005,
                'Rey_err': 2e-6,
           'Rmi': 0.030401,
           'Rmi_err': 2e-6,
           'Ly': 0.031,
           'Ly_err': 2e-6,
           #'p1': 0.0015,
           #'d1': 0.0015,
           #'d4': 0.004,
           'd3': 0.003,
           'd3_err': 2e-6,
           'Rp': 0.0347005,
           'Rp_err': 2e-6,
           'Rx': 0.035305,
           'Rx_err': 2e-6,
           'Rphi': 0.0353,
           'Rphi_err': 2e-6,
           'h': 0.007915, #Lx in MIC-DC-S-7-TS-SU-7350-ONE-1.pdf
           'h_err': 2e-6,
           'Lphi': 0.056,
            'Lphi_err': 2e-6}

is1_suref = {'Rey': 0.0148,
                 'Rey_err': 2e-6,
           'Rmi': 0.0154,
           'Rmi_err': 2e-6,
           'Ly': 0.015,
           'Ly_err': 2e-6,
           #'p1': 0.0015,
           #'d1': 0.0015,
           #'d4': 0.004,
           'd3': 0.003,
           'd3_err': 2e-6,
           'Rp': 0.019695,
           'Rp_err': 2e-6,
           'Rx': 0.0203,
           'Rx_err': 2e-6,
           'Rphi': 0.0203,
           'Rphi_err': 2e-6,
           'h': 0.008165, #Lx in MIC-DC-S-7-TS-SU-7350-ONE-1.pdf
           'h_err': 2e-6,
           'Lphi': 0.019,
            'Lphi_err': 2e-6}

is2_suref = {'Rey': 0.0298,
                 'Rey_err': 2e-6,
           'Rmi': 0.0304,
           'Rmi_err': 2e-6,
           'Ly': 0.031,
           'Ly_err': 2e-6,
           #'p1': 0.0015,
           #'d1': 0.0015,
           #'d4': 0.004,
           'd3': 0.003,
           'd3_err': 2e-6,
           'Rp': 0.0347,
           'Rp_err': 2e-6,
           'Rx': 0.0353,
           'Rx_err': 2e-6,
           'Rphi': 0.0353,
           'Rphi_err': 2e-6,
           'h': 0.007915, #Lx in MIC-DC-S-7-TS-SU-7350-ONE-1.pdf
           'h_err': 2e-6,
           'Lphi': 0.056,
            'Lphi_err': 2e-6}

def infiniteCylinderCapacitance(Rey, Rmi, Ly, taylor = False):
    """
    Compute capacitance of an infinite cylindrical capacitor
    """
    if not taylor:
        return 2*np.pi * epsilon_0 * Ly / np.log(Rmi / Rey)
    else:
        e = Rmi - Rey
        return 2*np.pi * epsilon_0 * Ly * (Rmi + Rey) / (2 * e)

def infiniteCylinderCapacitanceWithDips(Rey, Rmi, Ly, d3, taylor = False):
    """
    Compute capacitance of an infinite cylindrical capacitor with 4 dips 
    """
    if not taylor:
        return (2*np.pi - 4 * d3 / Rey) * epsilon_0 * Ly / np.log(Rmi / Rey)
    else:
        e = Rmi - Rey
        return (2*np.pi - 4 * d3 / Rey) * epsilon_0 * Ly * (Rmi + Rey) / (2 * e)


def stiffness(Rey, Rmi, Ly, d3, Rp, Rx, h, Rphi, Lphi, Vp, Vd, Vppyz, Vppx, Vppphi, verbose = False):
    k1 = yStiffness(Rey, Rmi, Ly, d3, Vp, Vd, Vppyz)
    k2 = zStiffness(Rey, Rmi, Ly, d3, Vp, Vd, Vppyz)
    k3 = xStiffness(Rp, Rx, h, Vp, Vd, Vppx)
    k4 = phiStiffness(Rp, Rphi, Lphi, Vp, Vd, Vppphi)
    k = k1 + k2 + k3 + k4
    if verbose:
        print('stiffness terms')
        print('   y', k1)
        print('   z', k2)
        print('   x', k3)
        print('   phi', k4)
    return k
        
    
def yStiffness(Rey, Rmi, Ly, d3, Vp, Vd, Vpp):
    alpha_y = 0
    e = Rmi - Rey
    S = ((2*np.pi * Rey) / 4 - d3) * Ly
    return 2 * epsilon_0 * S / e**3 * (1 + np.sinc(alpha_y))* ((Vpp - Vp)**2 + Vd**2)

def zStiffness(Rey, Rmi, Ly, d3, Vp, Vd, Vpp):
    alpha_z = np.pi/2
    e = Rmi - Rey
    S = ((2*np.pi * Rey) / 4 - d3) * Ly
    return 2 * epsilon_0 * S / e**3 * (1 - np.sinc(alpha_z))* ((Vpp - Vp)**2 + Vd**2)

def xStiffness(Rp, Rx, h, Vp, Vd, Vpp):
    e = Rx - Rp
    return np.pi * epsilon_0 * (Rx + Rp) * h / e**3 * ((Vpp - Vp)**2 + Vd**2)

def phiStiffness(Rp, Rphi, Lphi, Vp, Vd, Vpp, divideByTwo = False):
    e = Rphi - Rp
    return np.pi * epsilon_0 * Rphi * Lphi / e**3 * ((Vpp - Vp)**2 + Vd**2)


def cmpStiffnessDistrib(_is = 1, _su = 'SUEP', mode = 'HRM', errors = False, nsamples = 100000, vp_error = 0.01, vd_error = 0.01, bias = -0.085, adhoc_relative_error = 0.03, adhoc_absolute_error = 1e-4, plotDistrib = True):
    """
    Main function. Computes stiffness along Y axis, together with errors via Monte Carlo of metrology uncertainties. Can compute stiffness for one sample with nsamples = 1.
    bias -- bias wrt finite elements simulations
    adhoc_relative_error -- adhoc error to add, to take into account potentially bad knowledge of bias
    """
    if mode == 'HRM':
        if _is == 1 and _su == 'SUEP':
            isi = is1_suep
            Vp = 5
            Vd = 5
            Vppyz = 2.5
            Vppx = -5
            Vppphi = -10
        elif _is == 2 and _su == 'SUEP':
            isi = is2_suep
            Vp = 5
            Vd = 5
            Vppyz = 2.5
            Vppx = 0
            Vppphi = -10
        elif _is == 1 and _su == 'SUREF':
            isi = is1_suref
            Vp = 5
            Vd = 5
            Vppyz = 2.5
            Vppx = -5
            Vppphi = -10
        elif _is == 2 and _su == 'SUREF':
            isi = is2_suref
            Vp = 5
            Vd = 5
            Vppyz = 0
            Vppx = -10
            Vppphi = -10
        else:
            raise ValueError('WTH?!?')
    elif mode == 'FRM':
        if _is == 1 and _su == 'SUEP':
            isi = is1_suep
            Vp = 42
            Vd = 1
            Vppyz = 0
            Vppx = 0
            Vppphi = 0
        elif _is == 2 and _su == 'SUEP':
            isi = is2_suep
            Vp = 42
            Vd = 1
            Vppyz = 0
            Vppx = 0
            Vppphi = 0
        elif _is == 1 and _su == 'SUREF':
            isi = is1_suref
            Vp = 42
            Vd = 1
            Vppyz = 0
            Vppx = 0
            Vppphi = 0
        elif _is == 2 and _su == 'SUREF':
            isi = is2_suref
            Vp = 42
            Vd = 1
            Vppyz = 0
            Vppx = 0
            Vppphi = 0
        else:
            raise ValueError('WTH?!?')
    else:
        raise ValueError('Bad mode. Must be HRM or FRM.')

    #for SUEP, read voltages in data (N/A for SUREF b/c capa break)
    if _su == 'SUEP':
        if _is == 1:
            si = 'IS1'
        else:
            si = 'IS2'
        VpY, VdY = getVpVd(_su, si, 'Y', mode, useConstants = [Vp, Vd])
        VpZ, VdZ = getVpVd(_su, si, 'Z', mode, useConstants = [Vp, Vd])
        Vp = VpY
        Vd = VdY
        #print(VpY, VdY)
        #print(VpZ, VdZ)

    if not errors:
        Rey = isi['Rey']
        Rmi = isi['Rmi']
        Ly = isi['Ly']
        d3 = isi['d3']
        Rp = isi['Rp']
        Rx = isi['Rx']
        h = isi['h']
        Rphi = isi['Rphi']
        Lphi = isi['Lphi']
    else:
        Vp += np.random.randn(nsamples) * vp_error
        Vd += np.random.randn(nsamples) * vd_error
        Vppyz += np.random.randn(nsamples) * vp_error
        Vppx += np.random.randn(nsamples) * vp_error
        Vppphi += np.random.randn(nsamples) * vp_error
        Rey = isi['Rey'] + np.random.randn(nsamples) * isi['Rey_err']
        Rmi = isi['Rmi'] + np.random.randn(nsamples) * isi['Rmi_err']
        Ly = isi['Ly'] + np.random.randn(nsamples) * isi['Ly_err']
        d3 = isi['d3'] + np.random.randn(nsamples) * isi['d3_err']
        Rp = isi['Rp'] + np.random.randn(nsamples) * isi['Rp_err']
        Rx = isi['Rx'] + np.random.randn(nsamples) * isi['Rx_err']
        h = isi['h'] + np.random.randn(nsamples) * isi['h_err']
        Rphi = isi['Rphi'] + np.random.randn(nsamples) * isi['Rphi_err']
        Lphi = isi['Lphi'] + np.random.randn(nsamples) * isi['Lphi_err']

    k = stiffness(Rey, Rmi, Ly, d3, Rp, Rx, h, Rphi, Lphi, Vp, Vd, Vppyz, Vppx, Vppphi, verbose = False)
    print("--> eStiffness.cmpStiffnessDistrib (mean, bias, 1+bias)", np.mean(k), bias, 1+bias)
    k *= (1 + bias)
    
    if errors:
        if plotDistrib:
            #histo
            nbins = 50
            bins = np.linspace(np.min(k), np.max(k), nbins)
            bins, hist = histOutline(k, bins)
            plt.plot(bins, hist)
            plt.xlabel('k IS' + str(_is) + '-' + _su)
            plt.ylabel('pdf')
            plt.show(block = True)
        k_mean = np.mean(k)
        k_err = np.std(k, ddof = 1)
        #plt.show(block = True)
    else:
        k_mean = k
        k_err = 0

    k_error = np.sqrt(k_err**2 + (adhoc_relative_error * k_mean)**2 + adhoc_absolute_error**2)
        
    return k_mean, k_error, k_err
    

def tst(_is = 1, _su = 'SUEP'):
    #if _is == 1:
    #    isi = is1
    #elif _is == 2:
    #    isi = is2
    #else:
    #    raise ValueError('Bad _is')
    #for taylor in [False, True]:
    #    icc = infiniteCylinderCapacitance(isi['Rey'], isi['Rmi'], isi['Ly'], taylor = taylor)
    #    iccwd = infiniteCylinderCapacitanceWithDips(isi['Rey'], isi['Rmi'], isi['Ly'], isi['d3'], taylor = taylor)
    #    print(icc, iccwd)

    if _is == 1 and _su == 'SUEP':
        isi = is1_suep
        Vp = 5
        Vd = 5
        Vppyz = 2.5
        Vppx = -5
        Vppphi = -10
    elif _is == 2 and _su == 'SUEP':
        isi = is2_suep
        Vp = 5
        Vd = 5
        Vppyz = 2.5
        Vppx = 0
        Vppphi = -10
    elif _is == 1 and _su == 'SUREF':
        isi = is1_suref
        Vp = 5
        Vd = 5
        Vppyz = 2.5
        Vppx = -5
        Vppphi = -10
    elif _is == 2 and _su == 'SUREF':
        isi = is2_suref
        Vp = 5
        Vd = 5
        Vppyz = 0
        Vppx = -10
        Vppphi = -10
    else:
        raise ValueError('WTH?!?')
        
    k1_DH = yStiffness(isi['Rey'], isi['Rmi'], isi['Ly'], isi['d3'], Vp, Vd, Vppyz)
    k2_DH = zStiffness(isi['Rey'], isi['Rmi'], isi['Ly'], isi['d3'], Vp, Vd, Vppyz)
    k3_DH = xStiffness(isi['Rp'], isi['Rx'], isi['h'], Vp, Vd, Vppx)
    k4_DH = phiStiffness(isi['Rp'], isi['Rphi'], isi['Lphi'], Vp, Vd, Vppphi)
    k_DH = k1_DH + k2_DH + k3_DH + k4_DH
    
    # print('Vd, Vp', Vd, Vp)
    # print('k (terms)', k1, k2, k3)
    # print('k1 (MR)', k1_MR)
    # print('k (total)', k)
    # print('k (terms) *4', 4*k1, 4*k2, 4*k3)
    # print('k (first 2 terms) *4', 4*k1 + 4*k2)
    # print('k (total) *4', 4*k)

    # print(_su, 'IS' + str(_is))
    # print('k (total)', k)
    # print('k (total) *4', 4*k)

    # print('MR/CQG')
    # print('terms:', k1_MR, k2_MR, k3_MR, k4_MR)
    # print('total:', k_MR)
    # print('total (kphi/=2)', k_MR2)
    # print('total ((kx+kphi)/=2)', k_MR3)
    # print('total (kx*=2)', k_MR4)
    # print('----> total/2:', k_MR/2)

    print('DH')
    print('terms:', k1_DH, k2_DH, k3_DH, k4_DH)
    print('total:', k_DH)
