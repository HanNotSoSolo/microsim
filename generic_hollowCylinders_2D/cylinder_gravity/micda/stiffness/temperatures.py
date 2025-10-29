"""
Temperature probes in SU are positioned as:
 - T1 and T2 on top of the SU, roughly along z-axis, with T1 at z<0 and T2 at z>0, with |z|~R, with R the radius of the SU
 - T4 and T6 on bottom of the SU, roughly along z-axis, with T6 at z>0 and T4 at z<0, with |z|~R, with R the radius of the SU
 - T3 and T5 at mid-height of the SU, at z>0 (probably the same z as T2 and T6), with the segment [T3,T5] parallel to y-axis, with T3 at y>0 and T5 at y<0, at |y|~R/2, with R the radius of the SU
"""

import numpy as np
import matplotlib.pyplot as plt
from  micda.daio import rdData
import os.path
from micda.stiffness.dataPath import getPath

def getData(session, _su, missingDataMngt = 'trim', plotit = False):
    if not _su in ['SUEP', 'SUREF']:
        raise ValueError("Bad SU", _su)

    #if session == 218:
    #    _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/Phase_3/Session_218_EPR_V3DFIS2_01_SUEP/N0c_01/"
    #else:
    #    raise NotImplementedError()
    _path = getPath(session, _su)

    Temperature1_file = os.path.join(_path, _su, 'Temperature1.bin')
    Temperature2_file = os.path.join(_path, _su, 'Temperature2.bin')
    Temperature3_file = os.path.join(_path, _su, 'Temperature3.bin')
    Temperature4_file = os.path.join(_path, _su, 'Temperature4.bin')
    Temperature5_file = os.path.join(_path, _su, 'Temperature5.bin')
    Temperature6_file = os.path.join(_path, _su, 'Temperature6.bin')
    TemperatureDET1_file = os.path.join(_path, _su, 'TemperatureDET1.bin')
    TemperatureDET2_file = os.path.join(_path, _su, 'TemperatureDET2.bin')

    t, T1, mask = rdData(Temperature1_file)
    t, T2, mask = rdData(Temperature2_file)
    t, T3, mask = rdData(Temperature3_file)
    t, T4, mask = rdData(Temperature4_file)
    t, T5, mask = rdData(Temperature5_file)
    t, T6, mask = rdData(Temperature6_file)
    t, TDet1, mask = rdData(TemperatureDET1_file)
    t, TDet2, mask = rdData(TemperatureDET2_file)
    t -= t[0]

    bd = np.where(mask == 0)[0]
    nbd = np.size(bd)
    if nbd > 0:
        if missingDataMngt == 'ignore':
            pass
        elif missingDataMngt == 'trim':
            t = t[mask == 1]
            T1 = T1[mask == 1]
            T2 = T2[mask == 1]
            T3 = T3[mask == 1]
            T4 = T4[mask == 1]
            T5 = T5[mask == 1]
            T6 = T6[mask == 1]
            TDet1 = TDet1[mask == 1]
            TDet2 = TDet2[mask == 1]
        elif missingDataMngt == 'global mean':
            T1[bd] = np.mean(T1[mask == 1])
            T2[bd] = np.mean(T2[mask == 1])
            T3[bd] = np.mean(T3[mask == 1])
            T4[bd] = np.mean(T4[mask == 1])
            T5[bd] = np.mean(T5[mask == 1])
            T6[bd] = np.mean(T6[mask == 1])
            TDet1[bd] = np.mean(TDet1[mask == 1])
            TDet2[bd] = np.mean(TDet2[mask == 1])
        else:
            raise NotImplementedError()

    T1 += 273.16 #K
    T2 += 273.16 #K
    T3 += 273.16 #K
    T4 += 273.16 #K
    T5 += 273.16 #K
    T6 += 273.16 #K
    TDet1 += 273.16 #K
    TDet2 += 273.16 #K
        
    if plotit:
        if missingDataMngt != 'trim':
            ax = plt.subplot(211)
        plt.plot(t, T1, label = 'T1')
        plt.plot(t, T2, label = 'T2')
        plt.plot(t, T3, label = 'T3')
        plt.plot(t, T4, label = 'T4')
        plt.plot(t, T5, label = 'T5')
        plt.plot(t, T6, label = 'T6')
        plt.plot(t, TDet1, label = 'TDet1')
        plt.plot(t, TDet2, label = 'TDet2')
        plt.legend()
        plt.xlabel('t [s]')
        plt.ylabel('Temperatures [K]')
        if missingDataMngt != 'trim':
            plt.subplot(212, sharex = ax)
            plt.plot(t, mask)
            plt.ylim(-0.1, 1.1)
            plt.ylabel('Mask')
            plt.xlabel('t [s]')
        plt.show()

    return t, T1, T2, T3, T4, T5, T6, TDet1, TDet2


def cmpTGradients(session, _su, missingDataMngt = 'trim'):
    t, T1, T2, T3, T4, T5, T6, TDet1, TDet2 = getData(session, _su)
    T_mid_height = 0.5 * (T3 + T5) #temperature at mid-height of the SU, roughly aligned with lower and upper thermal probes; take mean, so we assume that temperatures varies more or less linearly between T3 and T5

    #plot vertical gradient as a function of time
    ax = plt.subplot(221)
    plt.plot(t, T6, label = 'Bottom (T6)')
    plt.plot(t, T_mid_height, label = 'Middle (mean(T3, T5))')
    plt.plot(t, T2, label = 'Top (T2)')
    plt.legend()
    plt.title('Vertical gradient')
    #plt.xlabel('t [s]')
    plt.ylabel('T [K]')

    #plot gradient along y (middle)
    plt.subplot(222, sharex = ax)
    plt.plot(t, T5, label = 'y<0 (T5)')
    plt.plot(t, T3, label = 'y>0 (T3)')
    plt.legend()
    plt.title('y-gradient (middle)')
    #plt.xlabel('t [s]')
    plt.ylabel('T [K]')

    #plot gradient along z (bottom)
    plt.subplot(223, sharex = ax)
    plt.plot(t, T4, label = 'z<0 (T4)')
    plt.plot(t, T6, label = 'z>0 (T6)')
    plt.legend()
    plt.title('z-gradient (bottom)')
    plt.xlabel('t [s]')
    plt.ylabel('T [K]')

    #plot gradient along z (top)
    plt.subplot(224, sharex = ax)
    plt.plot(t, T1, label = 'z<0 (T1)')
    plt.plot(t, T2, label = 'z>0 (T2)')
    plt.legend()
    plt.title('z-gradient (top)')
    plt.xlabel('t [s]')
    plt.ylabel('T [K]')
    plt.tight_layout()
    plt.show()

def tst(session = 218, _su = 'SUEP'):
    t, T1, T2, T3, T4, T5, T6, TDet1, TDet2 = getData(session, _su, plotit = True)
