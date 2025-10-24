#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 18:11:19 2025

@author: mdellava
"""

import gc

import numpy as np
import matplotlib.pyplot as plt

from spheres_2D import ForceOnTwoSpheres as FO2S

"""
Here the objective is to test the gravitational Yukawa's potential for two
uniform spheres separated by a distance d. Alpha and lambda parameters,
interveining in Yukawa's potential definition, will also be declared.
Apart from only studying different distances between spheres, we also pick some
values for lambda parameter in order to observe the evolution of the obtained
precision. This one will be calculated using the equations given by the
Adelberger paper, 2003.
"""

''' === VARIABLES DEFINITION ==='''
# All the values are given in standard units
# First sphere
R_1 = 2  # [m]
rho_1 = 19972.  # [kg/m^3]

# Second sphere
R_2 = 4
rho_2 = 4278.

# Miscellaneous
FILENAME = 'spheres_2D'

# Distance between the spheres
d_i = 9  # initial distance
d_f = 20  # final distance
n_steps = 10  # number of steps

# Yukawa parameters
alpha = 0.5
lmbda_list = np.array([40, 20, 10, 5, 2.5, 1.25, 0.75, 0.375])
L_0 = 1  # FIXME L_0 =! 1 makes the problem explode
rho_0 = 10e4

# Physical parameters
DIM = 2
COORSYS = 'cylindrical'
SOLVER = 'ScipyDirect'

# Mesh size
minSize = 0.001
maxSize = 0.3

''' === END OF VARIABLES DECLARATION ==='''

# The distances that will be calculated
d = np.linspace(d_i, d_f, n_steps)

# Results array
F_fem = np.zeros((n_steps, 2, len(lmbda_list)))  # first column with force on sphere 1, second with force on sphere 2
F_ana = np.zeros((n_steps, len(lmbda_list)))  # analytical force
epsilon = np.zeros((n_steps, len(lmbda_list)))  # precision

# Starting the calculation iterations
for i in range(n_steps):
    # Generation of a two-spheres problem
    system = FO2S('spheres_2D', R_1=R_1, rho_1=rho_1, R_2=R_2, rho_2=rho_2,
                  d=d[i], minSize=minSize, maxSize=maxSize, dim=DIM)

    # Generation of the generic meshes
    mesh_int, mesh_ext = system.mesh_generation()

    # index that indicates where I am in lambda_list vector
    j = 0

    for lmbda in lmbda_list:

        print("--- NEW LAMBDA ---")
        # Resolution of the linear Klein-Gordon problem
        result_pp_yukawa = system.get_yukawa_potential(mesh_int, mesh_ext,
                                                       alpha=alpha,
                                                       lmbda=lmbda, L_0=L_0,
                                                       rho_0=rho_0)

        # Postprocessing on the results file to extract the force
        F_fem[i, 0, j], F_fem[i, 1, j], F_ana[i, j], epsilon[i, j] = system.postprocess_force(result_pp_yukawa,
                                                                                              alpha=alpha,
                                                                                              lmbda=lmbda,
                                                                                              rho_0=rho_0,
                                                                                              getYukawa=True)
        # Index that indicates where I am in lmbda_list vector
        j += 1

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()


# Creating the figure for the plots
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

# Drawing figures
for i in range(len(lmbda_list)):
    axs[0].semilogy(d, -F_fem[:, 0, i])
    axs[1].semilogy(d, epsilon[:, i])


# Adding all the features that could explain the graphic to my future self
# (and obv also make it look a bit less confusing for people)
axs[0].set_title('Yukawa gravitational force depending on $\lambda$')

axs[1].legend(lmbda_list, fontsize='small', title='$\lambda$ values')

axs[0].set_ylabel('$F_Y ~[N]$')
axs[1].set_ylabel('Ɛ')
axs[1].set_xlabel('$d ~[m]$')

axs[1].set_ylim(top=2)

axs[0].grid()
axs[1].grid()

plt.savefig("Spheres_2D_Yukawa.png", bbox_inches='tight')
