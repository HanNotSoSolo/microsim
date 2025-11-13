"""
Runs under python 2.7
Uses an ascii file created by k_analysis.combineAlphas to get MICROSCOPE's stiffness bounds on Yukawa, and pickle files (python 2.7) obtained with cyd.microscope.cpyPublishedLines to get published constraints
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

def plotLimits(limitsFile = 'stiffnessYukawa.dat', xlim = [1e-4, 1], ylim = [1e-4, 1e7], save2disk = False):
    cols = ['lmbda', 'limit']
    data = np.genfromtxt(limitsFile, names = cols)
    lmbda = data['lmbda']
    abs_upper_bound = data['limit']
    
    dpath = '/Users/jberge/science/codes/cyd/cyd/microscope/data/stiffness'
    hoskins = get_pickle_lines(os.path.join(dpath, 'hoskins.p'))
    lake = get_pickle_lines(os.path.join(dpath, 'lake.p'))
    hust20 = get_pickle_lines(os.path.join(dpath, 'hust20.p'))
    hust12 = get_pickle_lines(os.path.join(dpath, 'hust12.p'))
    ew07 = get_pickle_lines(os.path.join(dpath, 'ew07.p'))
        
    plt.plot(hoskins[0], hoskins[1], color = 'blue', linewidth = 2, linestyle = '-', label = 'Irvine')
    plt.fill_between(hoskins[0], hoskins[1], ylim[1], facecolor = 'lightyellow', linewidth = 0)
    plt.fill_between([hoskins[0][-1], xlim[1]], hoskins[1][-1], ylim[1], facecolor = 'lightyellow', linewidth = 0)

    #plt.plot(lake[0], lake[1], color = 'cyan', linewidth = 2, linestyle = '-', label = 'geophysical')
    #plt.fill_between(lake[0], lake[1], ylim[1], facecolor = 'yellow', linewidth = 0)
    #plt.fill_between([lake[0][-1], xlim[1]], lake[1][-1], ylim[1], facecolor = 'yellow', linewidth = 0)

    plt.plot(ew07[0], ew07[1], color = 'green', linewidth = 2, linestyle = '-', label = r'E${\rm \"{o}}$t-Wash 2007')
    plt.fill_between(ew07[0], ew07[1], ylim[1], facecolor = 'lightyellow', linewidth = 0)
    plt.fill_between([ew07[0][-1], xlim[1]], ew07[1][-1], ylim[1], facecolor = 'lightyellow', linewidth = 0)
    
    plt.plot(hust12[0], hust12[1], color = 'orange', linewidth = 2, linestyle = '-', label = 'HUST 2012')
    plt.fill_between(hust12[0], hust12[1], ylim[1], facecolor = 'lightyellow', linewidth = 0)
    plt.fill_between([hust12[0][-1], xlim[1]], hust12[1][-1], ylim[1], facecolor = 'lightyellow', linewidth = 0)
    
    plt.plot(hust20[0], hust20[1], color = 'red', linewidth = 2, linestyle = '-', label = 'HUST 2020')
    plt.fill_between(hust20[0], hust20[1], ylim[1], facecolor = 'lightyellow', linewidth = 0)
    plt.fill_between([hust20[0][-1], xlim[1]], hust20[1][-1], ylim[1], facecolor = 'lightyellow', linewidth = 0)
    
    plt.plot(lmbda, abs_upper_bound, color = 'black', linewidth = 2, label = 'MICROSCOPE')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$ [m]', fontsize = 17)
    plt.ylabel(r'|$\alpha$|', fontsize = 17)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 17)
    plt.fill_between(lmbda, abs_upper_bound, ylim[1], facecolor = 'yellow')
            
    plt.legend(loc = 'best', prop = {'size': 12})
    plt.text(1e6, 2e-6, 'Excluded region')
    plt.tight_layout()
    
    if not save2disk:
        plt.show()
    else:
        plt.savefig('fig_yukawa.png')


def get_pickle_lines(filename):
    if os.path.isfile(filename):
        pfile = open(filename, 'rb')
        c = pickle.load(pfile)
        pfile.close()
        lines = c['lines']
        if len(lines.keys()) > 1:
            raise ValueError('Line dictionary must contain only one line!')
        l = lines.keys()[0]
        return (lines[l]['xs'][2:], lines[l]['ys'][2:])
    else:
        print("WARNING! Could not get bounds from figure.")
        return None
