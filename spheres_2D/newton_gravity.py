#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:03:31 2025

@author: mdellava
"""

import numpy as np
import matplotlib.pyplot as plt

from spheres_2D import ForceOnTwoSpheres as FO2S

"""
In this use case, we want to test the gravitational Newton's potential for two
uniform spheres separated by a distance d. This parameter is the only one that
will change along the experiment, since all other parameters will remain the
same. We will also challenge (a bit) the computer, as the demanded precision
will be at least 10e-6.
This script will run a hundred of measures and plot them alongside the
precision reached.
"""

''' === VARIABLES DEFINITION ==='''
# All the values are given in standard units
# First sphere
R_1 = 2  # [m]
rho_1 = 19972.  # [kg/m^3]
rho_q_1 = 19e-8  # [C/m^3]

# Second sphere
R_2 = 4
rho_2 = 4278.
rho_q_2 = -19e-8

# Miscellaneous
FILENAME = 'spheres_2D'

# Distance between the spheres
d_i = 9  # initial distance
d_f = 20  # final distance
n_steps = 20  # number of steps

# Physical parameters
DIM = 2
COORSYS = 'cylindrical'
SOLVER = 'ScipyDirect'

# Mesh size
minSize = 0.001
maxSize = 0.1
''' === END OF VARIABLES DECLARATION ==='''

# The distances that will be calculated
d = np.linspace(d_i, d_f, n_steps)

# Results array
F_fem = np.zeros((n_steps, 2))  # first column with force on sphere 1, second with force on sphere 2
F_ana = np.zeros((n_steps))  # analytical force
epsilon = np.zeros((n_steps, 2))  # precision

# Starting the calculation iterations
for i in range(n_steps):

    # Generation of a two-spheres problem
    system = FO2S('spheres_2D', R_1=R_1, rho_1=rho_1, R_2=R_2, rho_2=rho_2,
                  d=d[i], minSize=minSize, maxSize=maxSize, dim=DIM)

    # Generation of the generic meshes
    mesh_int, mesh_ext = system.mesh_generation()

    # Resolution of the Poisson's problem
    result_pp_newton = system.get_newton_force(mesh_int, mesh_ext)

    # Postprocessing on the results file to extract the force
    F_fem[i, 0], F_fem[i, 1], epsilon[i, 0] = system.postprocess_force(result_pp_newton, getNewton=True)

    # Finding analytical force
    F_ana[i] = -6.6743e-11 * ((4/3) * np.pi)**2 * (R_1**3 * rho_1 * R_2**3 * rho_2) / d[i]**2

    # Precision on second sphere
    epsilon[i, 1] = np.abs((F_ana[i] + F_fem[i, 1]) / F_ana[i])

mepsilon = np.array([np.mean(epsilon[:, 0]), np.mean(epsilon[:, 1])])


ax1 = plt.subplot(221)
ax1.plot(d, F_fem[:, 0], color='green')
ax1.plot(d, F_ana, color='blue')
ax1.legend(['FEM analysis', 'Analytical calculation'], fontsize='small')
ax1.set_title('Force on $S_1$')
ax1.set_ylabel('$F_r ~[N]$')

ax2 = plt.subplot(222)
ax2.plot(d, F_fem[:, 1], color='red')
ax2.plot(d, -F_ana, color='blue')
ax2.legend(['FEM analysis', 'Analytical calculation'], fontsize='small')
ax2.set_title('Force on $S_2$')

ax3 = plt.subplot(212)
ax3.semilogy(d, epsilon[:, 0], color='white')  # will not be seen
ax3.semilogy(d, epsilon[:, 1], color='white')  # will not be seen
ax3.set_xlabel('$d ~[m]$')
ax3.set_ylabel('$ε$')
twin_ax = ax3.twinx()
twin_ax.semilogy(d, epsilon[:, 0], color='green')
twin_ax.semilogy(d, epsilon[:, 1], color='red')
twin_ax.axhline(y=np.mean(epsilon[:, 0]), ls='--', color='green')
twin_ax.axhline(y=np.mean(epsilon[:, 1]), ls='--', color='red')
twin_ax.legend(['$ε_1$', '$ε_2$'], fontsize='small')
twin_ax.set_yticks(mepsilon, [f"{mepsilon[0]:.3e}", f"{mepsilon[1]:.3e}"])

plt.savefig("Spheres_2D_Newton.png", bbox_inches='tight')
