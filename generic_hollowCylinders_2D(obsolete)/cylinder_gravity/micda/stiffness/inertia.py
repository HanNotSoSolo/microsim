import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os.path
from scipy.interpolate import interp1d
from micda.daio import rdData
from micda.stiffness.rdStiffnessData import inertia
from micda.stiffness.dataPath import getPath

    
def cmpInertiaAcc(_su = 'SUEP', _is = 'IS1', _axis = 'Y', mode = 'HRM'):
    """
    The pair (_su, mode) sets which session must be used
    """
    clin = inertia(_su, _is, _axis, mode)
    clin.cmp_inertiaMatrix()
    print(clin.getOmegaX())
    print(clin.inMatrix_xx)
    clin.pltOmegas()
    clin.pltIn()
    clin.pltInAcc()
    clin.pltInAcc(denoise = True)
    

def compAPID108_oramic():
    """
    Compare inertia matrix from APID108 (VitesseAngulaire.bin etc) and ORAMIC (MIC_CECT_PRECISE_ATTITUDE), to see of ORAMIC provides lower noise.
    For stiffness sessions, we don't have ORAMIC attitude, but it will be helpful to know if we can decrease the noise.
    """
    clin = inertia('SUEP', 'IS1', 'X', 'HRM', forcePath = '/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/Phase_3/Session_218_EPR_V3DFIS2_01_SUEP/N0c_01')
    clin.rdOramic()
    clin.pltInAcc(use_oramic = False, xlim = [0, 200000], correct4mean = True)
    clin.pltInAcc(use_oramic = True, xlim = [0, 200000], correct4mean = True)
    
    
