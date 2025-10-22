import numpy as np
import k_mcmc

def writeNorm(norm, noBrackets = False):
    if not noBrackets:
        if norm == 1e2:
            return " [$\\times10^{-2}$]"
        elif norm == 1e3:
            return " [$\\times10^{-3}$]"
        elif norm == 1e4:
            return " [$\\times10^{-4}$]"
        elif norm == 1e5:
            return " [$\\times10^{-5}$]"
        elif norm == 1e6:
            return " [$\\times10^{-6}$]"
        elif norm == 1e7:
            return " [$\\times10^{-7}$]"
        elif norm == 1e8:
            return " [$\\times10^{-8}$]"
        elif norm == 1e9:
            return " [$\\times10^{-9}$]"
        else:
            raise NotImplementedError()
    else:
        if norm == 1e2:
            return " $\\times10^{-2}$"
        elif norm == 1e3:
            return " $\\times10^{-3}$"
        elif norm == 1e4:
            return " $\\times10^{-4}$"
        elif norm == 1e5:
            return " $\\times10^{-5}$"
        elif norm == 1e6:
            return " $\\times10^{-6}$"
        elif norm == 1e7:
            return " $\\times10^{-7}$"
        elif norm == 1e8:
            return " $\\times10^{-8}$"
        elif norm == 1e9:
            return " $\\times10^{-9}$"
        else:
            raise NotImplementedError()

def writeLatexTable(correct4viscousDamping = False, e_bias = -0.085, e_adhoc_relative_error = 0.03, k_norm = None, kgw_norm = None, kexc_norm = None, kN_norm = None, delta_norm = None, kl_norm = None, measuredContributionsOnly = True, comparisonsOnly = False):
    if measuredContributionsOnly and comparisonsOnly:
        raise ValueError("measuredContributionsOnly and comparisonsOnly are exclusive... One at a time... or none...")
    
    if k_norm is None:
        k_norm = 1e2
    if kgw_norm is None:
        if not correct4viscousDamping:
            kgw_norm = 1e4
        else:
            kgw_norm = 1e7
    if kexc_norm is None:
        kexc_norm = 1e4
    if kN_norm is None:
        kN_norm = 1e8
    if delta_norm is None:
        delta_norm = 1e2
    if kl_norm is None:
        #kl_norm = 1e7
        kl_norm = 1e4


    with open('est_stiffness.tex', 'w') as ff:
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
        ff.writelines("\\input{est_stiffness_table.tex}\n")
        if not measuredContributionsOnly and not comparisonsOnly:
            ff.writelines("\\end{landscape}\n")
        ff.writelines("\\end{document}\n")
        ff.close()
        
    with open('est_stiffness_table.tex', 'w') as f:
        f.writelines("\\begin{table}%[htdp]\n")
        if not comparisonsOnly:
            if not correct4viscousDamping:
                f.writelines("\\caption{Estimated model parameters when ignoring gold wire's viscous damping. (1): in-phase with position, corrected for excitation. (2): out-of-phase. All units but $Q$'s in N/m.} \n")
            else:
                f.writelines("\\caption{Estimated model parameters when considering gold wire's viscous damping. (1): in-phase with position, corrected for excitation. (2): out-of-phase. All units but $Q$'s in N/m.} \n")
        else:
            f.writelines("\\caption{Estimated (on-phase) stiffness and comparison with theory. All units but $Q$'s in N/m.} \n")
        f.writelines("\\small\n")
        f.writelines("\\begin{center}\n")

        if not measuredContributionsOnly and not comparisonsOnly:
            if not correct4viscousDamping:
                f.writelines("\\begin{tabular}{cc|cccc||ccc}\n")
                f.writelines("\\hline\n")
                f.writelines("Sensor & Axis (mode) & $\hat{k}^{(1)}$ " + writeNorm(k_norm) + " & $k_w^{(2)}$" + writeNorm(kgw_norm) + " & $Q$ & $k_w/Q$" + writeNorm(kgw_norm) + " & $k_\epsilon$" + writeNorm(k_norm) + " & $k_N$ & $\Delta k$" + writeNorm(delta_norm) + " \\\\\n")
            else:
                f.writelines("\\begin{tabular}{cc|ccccc||ccc}\n")
                f.writelines("\\hline\n")
                f.writelines("Sensor & Axis (mode) & $\hat{k}^{(1)}$" + writeNorm(k_norm) + " & $\lambda_w\omega$" + writeNorm(kl_norm) + " & $k_w^{(2)}$" + writeNorm(kgw_norm) + " & $Q$ & $k_w/Q$" + writeNorm(kgw_norm) + " & $k_\epsilon$" + writeNorm(k_norm) + " & $k_N$ & $\Delta k$" + writeNorm(delta_norm) + " \\\\\n")
        elif measuredContributionsOnly:
            if not correct4viscousDamping:
                f.writelines("\\begin{tabular}{cc|cccc}\n")
                f.writelines("\\hline\n")
                f.writelines("Sensor & Axis (mode) & $\hat{k}^{(1)}$ " + writeNorm(k_norm) + " & $k_w^{(2)}$" + writeNorm(kgw_norm) + " & $Q$" + writeNorm(kgw_norm) + " & $k_w/Q$" + writeNorm(kgw_norm) + " \\\\\n")
            else:
                f.writelines("\\begin{tabular}{cc|ccccc}\n")
                f.writelines("\\hline\n")
                f.writelines("Sensor & Axis (mode) & $\hat{k}^{(1)}$" + writeNorm(k_norm) + " & $\lambda_w \omega$" + writeNorm(kl_norm) + " & $k_w^{(2)}$" + writeNorm(kgw_norm) + " & $Q$ & $k_w/Q$" + writeNorm(kgw_norm) + " \\\\\n")
        elif comparisonsOnly:
            f.writelines("\\begin{tabular}{cc|cccc}\n")
            f.writelines("\\hline\n")
            f.writelines("Sensor & Axis (mode) & $\hat{k}^{(1)}$" + writeNorm(k_norm) + " & $k_\epsilon$" + writeNorm(k_norm) + " & $k_N$ & $\Delta k$" + writeNorm(delta_norm) + " \\\\\n")
        else:
            raise ValueError('?!?')
            
        f.writelines("\\hline\n")

        _su = ['SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUEP', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF', 'SUREF']
        _is = ['IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2', 'IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS1', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2', 'IS2']
        _ax = ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z']
        _mode = ['HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM', 'HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM', 'HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM', 'HRM', 'HRM', 'HRM', 'FRM', 'FRM', 'FRM']
        for i in range(24):
            mkLatexLine(f, _su[i], _is[i], _ax[i], _mode[i], correct4exc = True, correct4viscousDamping = correct4viscousDamping, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, k_norm = k_norm, kgw_norm = kgw_norm, kexc_norm = kexc_norm, kN_norm = kN_norm, delta_norm = delta_norm, kl_norm = kl_norm, measuredContributionsOnly = measuredContributionsOnly, comparisonsOnly = comparisonsOnly)
            if i in [5, 11, 17]:
                f.writelines("\\hline\n")
            if i == 11:
                f.writelines("\\hline\n")
            
        f.writelines("\\hline\n")
        f.writelines("\\end{tabular}\n")
        f.writelines("\\end{center}\label{tab_results1}\n")
        f.writelines("\\label{default}\n")
        f.writelines("\\end{table}%\n")

        

def mkLatexLine(f, _su, _is, _axis, mode, correct4exc = True, correct4viscousDamping = False, e_bias = -0.085, e_adhoc_relative_error = 0.03, k_norm = 1e3, kgw_norm = 1e4, kexc_norm = 1e4, kN_norm = 1e8, delta_norm = 1e4, kl_norm = 1e3, measuredContributionsOnly = True, comparisonsOnly = False):
    """
    line of the type: SUREF-IS1 & $X$ (HRM) & k & kw & Q & phi & k/Q & kexc & keps & kN & Dk if $X$ (HRM) or same thing with first column empty
    """
    if measuredContributionsOnly and comparisonsOnly:
        raise ValueError("measuredContributionsOnly and comparisonsOnly are exclusive... One at a time... or none...")
    
    d, params = k_mcmc.k_fits(_su = _su, _is = _is, _axis = _axis, mode = mode, correct4exc = True, correct4viscousDamping = correct4viscousDamping, forceFreq = True, e_bias = e_bias, e_adhoc_relative_error = e_adhoc_relative_error, plotit = False)
    k = k_norm * params['kE']
    kexc = kexc_norm * params['kexc']
    lambda_gw = kl_norm * params['lambda_gw'] #* np.sqrt(kexc) #multiply by omega_exc to be in N/m and allow for an easy comparison
    print("------> ", params['lambda_gw'] * np.sqrt(kexc))
    gw_k = kgw_norm * params['gw_k']
    gw_Q = params['gw_Q']
    gw_kQ = kgw_norm * params['k/Q']
    phi = params['phi']
    kN = kN_norm * params['kN']
    th_kE = k_norm * params['th_kE']
    Deltak = delta_norm * params['Deltak']

    k0 = k.nominal_value
    kerr = k.std_dev
    if correct4viscousDamping:
        lambda_gw0 = lambda_gw.nominal_value
        lambda_gw_err = lambda_gw.std_dev
    gw_k0 = gw_k.nominal_value
    gw_kerr = gw_k.std_dev
    gw_Q0 = gw_Q#.nominal_value
    gw_Qerr = 0#gw_Q.std_dev
    gw_kQ0 = gw_kQ.nominal_value
    gw_kQerr = gw_kQ.std_dev
    th_kE0 = th_kE.nominal_value
    th_kEerr = th_kE.std_dev
    Delta_k0 = Deltak.nominal_value
    Delta_kerr = Deltak.std_dev

    if _axis == 'X' and mode == 'HRM':
        line = _su + " & $X$ (HRM) & "
    elif _axis == 'Y' and mode == 'HRM':
        line = _is + " & $Y$ (HRM) & "
    else:
        line = " & $" + _axis + "$ ("+ mode + ") & "

    line += "%0.3f" % k0 + "$\pm$" + "%0.3f" % kerr
    if not comparisonsOnly:
        if correct4viscousDamping:
            line += " & %0.3f" % lambda_gw0 + "$\pm$" + "%0.3f" % lambda_gw_err + " & "
            line += "%0.3f" % gw_k0 + "$\pm$" + "%0.3f" % gw_kerr + " & "
            line += "%0.1f" % gw_Q0 + "$\pm$" + "%0.1f" % gw_Qerr + " & "
            line += "%0.2f" % gw_kQ0 + "$\pm$" + "%0.2f" % gw_kQerr
        else:
            line += "& %0.3f" % gw_k0 + "$\pm$" + "%0.3f" % gw_kerr + " & "
            line += "%0.2f" % gw_Q0 + "$\pm$" + "%0.2f" % gw_Qerr + " & "
            line += "%0.2f" % gw_kQ0 + "$\pm$" + "%0.2f" % gw_kQerr
    if not measuredContributionsOnly:
        line += " & %0.3f" % th_kE0 + "$\pm$" + "%0.3f" % th_kEerr + " & "
        line += "%0.2f" % kN + writeNorm(kN_norm, noBrackets = True) + " & "
        line += "%0.3f" % Delta_k0 + "$\pm$" + "%0.3f" % Delta_kerr
    line += " \\\\\n"
    f.writelines(line)
    #return line
