#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:39:30 2025

@author: mdellava
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:39:38 2025

@author: mdellava
"""

import numpy as np
import matplotlib.pyplot as plt

from parallel_hollow_cylinders_2D import ForceOnTwoParallelCylinders as FO2PC

"""
This script will evaluate Yukawa gravitational force between two parallel
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

# Yukawa parameters
alpha = 1  # since it's only a scale parameter, it's not really important: we know it works
lmbda = np.array([R_ext_1/10, R_ext_1/2, 5*R_ext_1, R_ext_2/10, R_ext_2/2, 5*R_ext_2])
rho_0 = 10e4

# Miscellaneous
FILENAME = 'parallel_hollow_cylinders_2D'
VERBOSE = 1
FEM_ORDER = 1

# Physical parameters
COORSYS = 'cylindrical'
SOLVER = 'ScipyDirect'

# Mesh size
minSize = 0.00001
maxSize = 0.0005
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
F_fem = np.zeros((n_steps, 2, len(lmbda)))  # first column with force on sphere 1, second with force on sphere 2
F_ana = np.zeros((n_steps, len(lmbda)))  # analytical force
epsilon = np.zeros((n_steps, len(lmbda)))  # precision

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

    # Solving the same geometry with different lambdas
    for j in range(len(lmbda)):
        # Solving the Poisson problem
        result_pp_newton = system_2hc.get_yukawa_potential(mesh_int, mesh_ext,
                                                           alpha=alpha,
                                                           lmbda=lmbda[j],
                                                           rho_0=rho_0, )

        # Post-processing of the solution to get the force and relative error
        F_fem[i, 0, j], F_fem[i, 1, j], F_ana[i, j], _, epsilon[i, j], _ = system_2hc.postprocess_force(result_pp_newton,
                                                                                                        getYukawa=True,
                                                                                                        alpha=alpha,
                                                                                                        lmbda=lmbda[j],
                                                                                                        rho_0=rho_0)

# Evaluating the average relative error for each cylinder
mepsilon = np.array([np.mean(epsilon[:, 0]), np.mean(epsilon[:, 1])])


# Creating the figure for the plots
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

# Drawing figures
for i in range(len(lmbda)):
    axs[0].semilogy(z, -F_fem[:, 0, i])
    axs[1].semilogy(z, epsilon[:, i])


# Adding all the features that could explain the graphic to my future self
# (and obv also make it look a bit less confusing for people)
axs[0].set_title('Yukawa gravitational force depending on $\lambda$')

axs[1].legend(lmbda, fontsize='small', title='$\lambda$ values')

axs[0].set_ylabel('$F_Y ~[N]$')
axs[1].set_ylabel('Ɛ')
axs[1].set_xlabel('$d ~[m]$')

axs[1].set_ylim(top=2)

axs[0].grid()
axs[1].grid()

plt.savefig("Spheres_2D_Yukawa.png", bbox_inches='tight', dpi=700)