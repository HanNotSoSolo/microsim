#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:39:38 2025

@author: mdellava
"""

import numpy as np
import matplotlib.pyplot as plt
import gc

from parallel_hollow_cylinders_2D import ForceOnTwoParallelCylinders as FO2PC

"""
This script will evaluate Newton gravitational force between two parallel
hollow cylinders via the Finite Element Analysis. A graphic will be drawn to
compare this value with the analytical computation, performed with the script
used for the MICROSCOPE mission.
"""

''' === VARIABLES DEFINITION === '''
# All measures are given in standard units
# First cylinder
R_int_1 = 15.4e-3  # [m]
R_ext_1 = 19.7e-3
h_1 = 43.37e-3
rho_1 = 19972  # [kg/m³]
rho_q_1 = 0  # [C/m³]

# Second cylinder
R_int_2 = 30.4e-3
R_ext_2 = 34.6975e-3
h_2 = 79.83e-3
rho_2 = 4420
rho_q_2 = 0

# Axial displacement
z_i = -1e-5
z_f = 1e-5
n_steps = 21

# Miscellaneous
FILENAME = 'parallel_hollow_cylinders_2D'
VERBOSE = 1
FEM_ORDER = 1

# Physical parameters
COORSYS = 'cylindrical'
SOLVER = 'ScipyDirect'

# Mesh size
minSize = 0.0001
maxSize = 0.001
''' === END OF VARIABLES DECLARATION === '''

# The steps where the computation will take place
z = np.linspace(z_i, z_f, n_steps)
r = np.zeros_like(z)

# Since the system would pass in a neutral position that would be bad for
# relative error computation, the neutral position is removed
z = np.delete(z, 10)
r = np.delete(r, 10)
n_steps -= 1

# Results array
F_fem = np.zeros((n_steps, 2))
F_ana = np.zeros_like(F_fem)
epsilon = np.zeros_like(F_fem)

# Calculation iterations
for i in range(n_steps):
    # Creating the system
    system_2hc = FO2PC(problemName=FILENAME, R_int_1=R_int_1, R_ext_1=R_ext_1,
                       Z_1=z[i], R_1=r[i], rho_1=rho_1, h_1=h_1, R_int_2=R_int_2,
                       R_ext_2=R_ext_2, h_2=h_2, rho_2=rho_2, minSize=minSize,
                       maxSize=maxSize, FEM_ORDER=FEM_ORDER, SOLVER=SOLVER,
                       VERBOSE=VERBOSE, coorsys=COORSYS)

    # Checking the system's geometry
    system_2hc.GEOMETRY_VERIFICATIONS()

    # Generation of the meshes
    mesh_int, mesh_ext = system_2hc.mesh_generation(SHOW_MESH=False)

    # Solving the Poisson problem
    result_pp_newton = system_2hc.get_newton_potential(mesh_int, mesh_ext)

    # Post-processing of the solution to get the force and relative error
    F_fem[i, 0], F_fem[i, 1], F_ana[i, 0], F_ana[i, 1], epsilon[i, 0], epsilon[i, 1] = system_2hc.postprocess_force(result_pp_newton,
                                                                                                                    getNewton=True)

    # Manually collecting garbage because Python cannot do it himself
    # NOTE: this is important for memory usage
    gc.collect()

# Evaluating the average relative error for each cylinder
mepsilon = np.array([np.mean(epsilon[:, 0]), np.mean(epsilon[:, 1])])


# Drawing plot
ax1 = plt.subplot(221)
ax1.plot(z, F_fem[:, 0], color='green')
ax1.plot(z, F_ana[:, 0], color='blue')
ax1.legend(['FEM analysis', 'Analytical calculation'], fontsize='small')
ax1.set_title('Force on $S_1$')
ax1.set_ylabel('$F_z ~[N]$')

ax2 = plt.subplot(222)
ax2.plot(z, F_fem[:, 1], color='red')
ax2.plot(z, F_ana[:, 1], color='blue')
ax2.legend(['FEM analysis', 'Analytical calculation'], fontsize='small')
ax2.set_title('Force on $S_2$')

ax3 = plt.subplot(212)
ax3.semilogy(z, epsilon[:, 0], color='white')  # will not be seen
ax3.semilogy(z, epsilon[:, 1], color='white')  # will not be seen
ax3.set_xlabel('$d ~[m]$')
ax3.set_ylabel('$ε$')
twin_ax = ax3.twinx()
twin_ax.semilogy(z, epsilon[:, 0], color='green')
twin_ax.semilogy(z, epsilon[:, 1], color='red')
twin_ax.axhline(y=np.mean(epsilon[:, 0]), ls='--', color='green')
twin_ax.axhline(y=np.mean(epsilon[:, 1]), ls='--', color='red')
twin_ax.legend(['$ε_1$', '$ε_2$'], fontsize='small')
twin_ax.set_yticks(mepsilon, [f"{mepsilon[0]:.3e}", f"{mepsilon[1]:.3e}"], minor=True)


plt.savefig("Hollow_cylinders_2D_Newton.png", bbox_inches='tight')