import numpy as np
import k_analysis
from uncertainties import ufloat
import os

kQ = {'X-SUEP': ufloat(1.1e-3, 0.2e-3),
          'Y-SUEP': ufloat(0.3e-3,0.5e-3),
          'Z-SUEP': ufloat(1.3e-3,2e-3),
          'X-SUREF': ufloat(0.07e-3, 0.03e-3),
          'Y-SUREF': ufloat(0.2e-3,0.3e-3),
          'Z-SUREF': ufloat(0.7e-3, 0.3e-3)}

def writeNorm(norm, unit = None, noBrackets = False):
    if not noBrackets:
        if norm == 1e2:
            sn = " [$\\times10^{-2}"
        elif norm == 1e3:
            sn = " [$\\times10^{-3}"
        elif norm == 1e4:
            sn = " [$\\times10^{-4}"
        elif norm == 1e5:
            sn = " [$\\times10^{-5}"
        elif norm == 1e6:
            sn = " [$\\times10^{-6}"
        elif norm == 1e7:
            sn = " [$\\times10^{-7}"
        elif norm == 1e8:
            sn = " [$\\times10^{-8}"
        elif norm == 1e9:
            sn = " [$\\times10^{-9}"
        else:
            raise NotImplementedError(norm)
        if unit is None:
            sn += "$]"
        else:
            sn += "~{\\rm " + unit + "}$]"
    else:
        if norm == 1e2:
            sn = " $\\times10^{-2}"
        elif norm == 1e3:
            sn = " $\\times10^{-3}"
        elif norm == 1e4:
            sn = " $\\times10^{-4}"
        elif norm == 1e5:
            sn = " $\\times10^{-5}"
        elif norm == 1e6:
            sn = " $\\times10^{-6}"
        elif norm == 1e7:
            sn = " $\\times10^{-7}"
        elif norm == 1e8:
            sn = " $\\times10^{-8}"
        elif norm == 1e9:
            sn = " $\\times10^{-9}"
        else:
            raise NotImplementedError(norm)
        if unit is None:
            sn += "$"
        else:
            sn += "~{\\rm " + unit + "}$"
    return sn

#def writeLatexTable(Q = ufloat(100, 50), e_bias = -0.085, e_adhoc_relative_error = 0.03, k_norm = None, kgw_norm = None, kkQ_norm = None, kN_norm = None, delta_norm = None, kl_norm = None, measuredContributionsOnly = False, comparisonsOnly = False, _dir = '/Users/jberge/Desktop/ktable'):
def writeLatexTable(e_bias = -0.085, e_adhoc_relative_error = 0.03, k_norm = None, kgw_norm = None, kkQ_norm = None, kN_norm = None, delta_norm = None, kl_norm = None, measuredContributionsOnly = False, comparisonsOnly = False, _dir = '/Users/jberge/Desktop/ktable'):
    if measuredContributionsOnly and comparisonsOnly:
        raise ValueError("measuredContributionsOnly and comparisonsOnly are exclusive... One at a time... or none...")
    if measuredContributionsOnly:
        tableShortName = 'table_stiffness.tex'
        tableLabel = 'tab_stiffness'
    elif comparisonsOnly:
        tableShortName = 'table_results.tex'
        tableLabel = 'tab_results'
    else:
        tableShortName = 'table_complete.tex'
        tableLabel = 'tab_results'
    tableName = os.path.join(_dir, tableShortName)
    
    if k_norm is None:
        k_norm = 1e2
    if kgw_norm is None:
        kgw_norm = 1e2
    if kkQ_norm is None:
        kkQ_norm = 1e5
    if kN_norm is None:
        kN_norm = 1e8
    if delta_norm is None:
        delta_norm = 1e2
    if kl_norm is None:
        #kl_norm = 1e7
        kl_norm = 1e2


    with open(os.path.join(_dir, 'est_stiffness.tex'), 'w') as ff:
        ff.writelines("\\documentclass[11pt]{article}\n")
        ff.writelines("\\usepackage{graphicx}\n")
        ff.writelines("\\usepackage{amssymb}\n")
        ff.writelines("\\usepackage{amsmath}\n")
        ff.writelines("\\usepackage{lscape}\n")
        ff.writelines("\\usepackage{caption}\n")
        ff.writelines("\\usepackage{subcaption}\n")
        ff.writelines("\\setlength{\\textwidth}{15cm}\n")
        ff.writelines("\\setlength{\\textheight}{23cm}\n")
        ff.writelines("\\setlength{\\oddsidemargin}{1cm}\n")
        ff.writelines("\\setlength{\\evensidemargin}{1cm}\n")
        ff.writelines("\\setlength{\\voffset}{-2cm}\n")
        ff.writelines("\\begin{document}\n")
        ff.writelines("\\title{MICROSCOPE stiffness}\n")
        ff.writelines("\\author{latex}\n")
        ff.writelines("%\\date{}\n")
        ff.writelines("\\maketitle\n")
        if not measuredContributionsOnly and not comparisonsOnly:
            ff.writelines("\\begin{landscape}\n")
        ff.writelines("\\input{" + tableShortName + "}\n")
        if not measuredContributionsOnly and not comparisonsOnly:
            ff.writelines("\\end{landscape}\n")
        ff.writelines("\\end{document}\n")
        ff.close()
        
    with open(tableName, 'w') as f:
        f.writelines("\\begin{table}%[htdp]\n")
        if not comparisonsOnly:
            f.writelines("\\caption{Estimated model parameters.} \n")
        else:
            f.writelines("\\caption{Estimated stiffness and comparison with theory.} \n")
        f.writelines("\\small\n")
        f.writelines("\\begin{center}\n")

        if not measuredContributionsOnly and not comparisonsOnly:
            f.writelines("\\begin{tabular}{cc|cccc||cc||c}\n")
            f.writelines("\\hline\n")
            f.writelines("Sensor & Axis (mode) & $\hat{k}_0$ & $\hat{k}_w$ & $\hat{Q}$ & $\hat{\lambda}_w$ & $k_\epsilon$ & $k_N$ & $\Delta k$ \\\\\n")
            f.writelines(" & & " + writeNorm(k_norm, unit = 'N/m') + " & " + writeNorm(kgw_norm, unit = 'N/m') + " & & " + writeNorm(kl_norm, unit = 'Ns/m') + " & " + writeNorm(k_norm, unit = 'N/m') + " & [$10^{-8}$ N/m] & " + writeNorm(delta_norm, unit = 'N/m') + " \\\\\n")
        elif measuredContributionsOnly:
            f.writelines("\\begin{tabular}{cc|cccc}\n")
            f.writelines("\\hline\n")
            f.writelines("Sensor & Axis (mode) & $\hat{k}_0$ " + writeNorm(k_norm) + " & $\hat{k}_w$" + writeNorm(kgw_norm) + " & $\hat{Q}$ & $\hat{\lambda}_w$" + writeNorm(kl_norm) + " \\\\\n")
        elif comparisonsOnly:
            f.writelines("\\begin{tabular}{cc|ccc||c}\n")
            f.writelines("\\hline\n")
            f.writelines("Sensor & Axis (mode) & $\hat{k}_0$" + writeNorm(k_norm) + " & $k_\epsilon$" + writeNorm(k_norm) + " & $k_N$ & $\Delta k$" + writeNorm(delta_norm) + " \\\\\n")
        else:
            raise ValueError('?!?')
            
        f.writelines("\\hline\n")

        #first estimation at Q
        for su in ['SUEP', 'SUREF']:
            for iis in ['IS1', 'IS2']:
                res = k_analysis.k_fit_radial(_su = su, _is = iis, kyQ = ufloat(0.3e-3,0.5e-3), kzQ = ufloat(1.3e-3,2e-3), taylorOrder = 4, phase = 'position', e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = 0, plotit = False)
                Q = res['Q']
                for mode in ['HRM', 'FRM']:
                    mkLatexLineSingle(f, su, iis, 'X', mode, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = 1e-4, k_norm = k_norm, kgw_norm = kgw_norm, kkQ_norm = kkQ_norm, kN_norm = kN_norm, delta_norm = delta_norm, kl_norm = kl_norm, measuredContributionsOnly = measuredContributionsOnly, comparisonsOnly = comparisonsOnly, Q = Q, kQ = kQ['X-' + su])
                    mkLatexLineRadial(f, su, iis, mode, kyQ = kQ['Y-' + su], kzQ = kQ['Z-' + su], e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = 0, k_norm = k_norm, kgw_norm = kgw_norm, kkQ_norm = kkQ_norm, kN_norm = kN_norm, delta_norm = delta_norm, kl_norm = kl_norm, measuredContributionsOnly = measuredContributionsOnly, comparisonsOnly = comparisonsOnly)
                    if mode == 'FRM':
                        f.writelines("\\hline\n")
            if su == 'SUEP':
                f.writelines("\\hline\n")
        
        # _su = ['SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF']
        # _is = ['IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2', 'IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2']
        # _ax = ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z']
        # _mode = ['HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM', 'HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM', 'HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM', 'HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM']
        # raise NotImplementedError("...")
        # for i in range(24):
        #     mkLatexLine(f, _su[i], _is[i], _ax[i], _mode[i], e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, k_norm = k_norm, kgw_norm = kgw_norm, kkQ_norm = kkQ_norm, kN_norm = kN_norm, delta_norm = delta_norm, kl_norm = kl_norm, measuredContributionsOnly = measuredContributionsOnly, comparisonsOnly = comparisonsOnly, Q = Q)
        #     if i in [5, 11, 17]:
        #         f.writelines("\\hline\n")
        #     if i == 11:
        #         f.writelines("\\hline\n")
            
        f.writelines("\\hline\n")
        f.writelines("\\end{tabular}\n")
        f.writelines("\\end{center}\label{" + tableLabel + "}\n")
        f.writelines("\\label{default}\n")
        f.writelines("\\end{table}%\n")

        

def mkLatexLineSingle(f, _su, _is, _axis, mode, e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 1e-4, k_norm = 1e3, kgw_norm = 1e4, kkQ_norm = 1e4, kN_norm = 1e8, delta_norm = 1e4, kl_norm = 1e3, measuredContributionsOnly = False, comparisonsOnly = False, Q = ufloat(100, 50), kQ = ufloat(1.1e-3, 0.2e-3)):
    """
    Results for single axis/mode estimation, with Q fixed
    line of the type: SUREF-IS1 & $X$ (HRM) & k & kw & Q & phi & k/Q & kexc & keps & kN & Dk if $X$ (HRM) or same thing with first column empty
    """
    if measuredContributionsOnly and comparisonsOnly:
        raise ValueError("measuredContributionsOnly and comparisonsOnly are exclusive... One at a time... or none...")
    
    #k0, kw, kQ, k_corr, kE_th, kN = k_analysis.k_fit(_su = _su, _is = _is, _axis = _axis, mode = mode, Q = Q, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, plotit = False)
    _res = k_analysis.k_fit(_su = _su, _is = _is, _axis = _axis, mode = mode, fit_kw = False, fit_lambda = True, kQ = kQ, Q = Q, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error, plotit = False)
    k0 = _res['k0']
    gw_k = _res['kw']
    lambda_gw = _res['lambda_v']
    k_corr = _res['k_corr']
    kE_th = _res['kE_th']
    kN = _res['kN']
    
    k = k_norm * k0
    k_corr *= k_norm
    kE_th *= k_norm
    lambda_gw *= kl_norm
    gw_k *= kgw_norm
    #gw_kQ = kkQ_norm * kQ
    gw_Q = Q
    kN *= kN_norm
    #th_kE = k_norm * params['th_kE']
    #Deltak = delta_norm * params['Deltak']

    k0 = k.nominal_value
    kerr = k.std_dev
    lambda_gw0 = lambda_gw.nominal_value
    lambda_gw_err = lambda_gw.std_dev
    gw_k0 = gw_k.nominal_value
    gw_kerr = gw_k.std_dev
    gw_Q0 = Q.nominal_value
    gw_Qerr = Q.std_dev
    kE_th0 = kE_th.nominal_value
    kE_therr = kE_th.std_dev
    Delta_k0 = k_corr.nominal_value
    Delta_kerr = k_corr.std_dev

    if _axis == 'X' and mode == 'HRM':
        line = _su + " & $X$ (HRM) & "
    elif _axis == 'Y' and mode == 'HRM':
        line = _is + " & $Y$ (HRM) & "
    else:
        line = " & $" + _axis + "$ ("+ mode + ") & "

    line += "%0.2f" % k0 + "$\pm$" + "%0.2f" % kerr
    if not comparisonsOnly:
        line += "& %0.2f" % gw_k0 + "$\pm$" + "%0.2f" % gw_kerr + " & "
        #line += "%0.3f" % gw_kQ0 + "$\pm$" + "%0.3f" % gw_kQerr
        line += "%0.1f" % gw_Q0 + "$\pm$" + "%0.1f" % gw_Qerr + " & "
        #line += "N/A & " #Q... fixed in this case
        line += "%0.2f" % lambda_gw0 + "$\pm$" + "%0.2f" % lambda_gw_err
    if not measuredContributionsOnly:
        line += " & %0.2f" % kE_th0 + "$\pm$" + "%0.2f" % kE_therr + " & "
        #line += "%0.2f" % kN + writeNorm(kN_norm, unit = None, noBrackets = True) + " & "
        line += "%0.2f" % kN + " & "
        line += "%0.3f" % Delta_k0 + "$\pm$" + "%0.3f" % Delta_kerr
    line += " \\\\\n"
    f.writelines(line)
    #return line


def mkLatexLineRadial(f, _su, _is, mode, kyQ = ufloat(0.3e-3,0.5e-3), kzQ = ufloat(1.3e-3,2e-3), e_bias = -0.085, e_adhoc_relative_error = 0.03, e_adhoc_absolute_error = 0, k_norm = 1e3, kgw_norm = 1e4, kkQ_norm = 1e4, kN_norm = 1e8, delta_norm = 1e4, kl_norm = 1e3, measuredContributionsOnly = False, comparisonsOnly = False):
    """
    Results for combined radial stiffness for a given sensor
    line of the type: SUREF-IS1 & $X$ (HRM) & k & kw & Q & phi & k/Q & kexc & keps & kN & Dk if $X$ (HRM) or same thing with first column empty
    """
    if not mode in ['HRM', 'FRM']:
        raise ValueError('Bad mode')
    if measuredContributionsOnly and comparisonsOnly:
        raise ValueError("measuredContributionsOnly and comparisonsOnly are exclusive... One at a time... or none...")
    
    res = k_analysis.k_fit_radial(_su = _su, _is = _is, kyQ = kyQ, kzQ = kzQ, taylorOrder = 4, phase = 'position', e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, e_adhoc_absolute_error = e_adhoc_absolute_error, plotit = False)
    for _ax in ['y', 'z']:
        k0 = res['k0_' + _ax + mode]
        k_corr = res['k_corr_' + _ax + mode]
        kE_th = res['kth_' + _ax + mode]
        lambda_gw = res['lambda_' + _ax + mode]
        gw_k = res['kw_' + _ax]
        gw_Q = res['Q']
        kN = res['kN']
        
        k = k_norm * k0
        k_corr *= k_norm
        kE_th *= k_norm
        lambda_gw *= kl_norm 
        gw_k *= kgw_norm
        kN *= kN_norm
        #th_kE = k_norm * params['th_kE']
        #Deltak = delta_norm * params['Deltak']

        k0 = k.nominal_value
        kerr = k.std_dev
        lambda_gw0 = lambda_gw.nominal_value
        lambda_gw_err = lambda_gw.std_dev
        gw_k0 = gw_k.nominal_value
        gw_kerr = gw_k.std_dev
        gw_Q0 = gw_Q.nominal_value
        gw_Qerr = gw_Q.std_dev
        kE_th0 = kE_th.nominal_value
        kE_therr = kE_th.std_dev
        Delta_k0 = k_corr.nominal_value
        Delta_kerr = k_corr.std_dev

        if _ax == 'y' and mode == 'HRM':
            line = _is + " & $Y$ (HRM) & "
        else:
            line = " & $" + _ax.upper() + "$ ("+ mode + ") & "

        line += "%0.2f" % k0 + "$\pm$" + "%0.2f" % kerr
        if not comparisonsOnly:
            line += "& %0.2f" % gw_k0 + "$\pm$" + "%0.2f" % gw_kerr + " & "
            line += "%0.1f" % gw_Q0 + "$\pm$" + "%0.1f" % gw_Qerr + " & "
            line += "%0.2f" % lambda_gw0 + "$\pm$" + "%0.2f" % lambda_gw_err
        if not measuredContributionsOnly:
            line += " & %0.2f" % kE_th0 + "$\pm$" + "%0.2f" % kE_therr + " & "
            #line += "%0.2f" % kN + writeNorm(kN_norm, unit = None, noBrackets = True) + " & "
            line += "%0.2f" % kN +  " & "
            line += "%0.3f" % Delta_k0 + "$\pm$" + "%0.3f" % Delta_kerr
        line += " \\\\\n"
        f.writelines(line)

