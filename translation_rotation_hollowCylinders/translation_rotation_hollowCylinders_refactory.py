#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Re)Created on Tue Nov  27 15:13:30 2025

@author: mdellava
"""

# Libraries and modules
# Core modules for file manipulation in the system
from shutil import rmtree
import gc

# Math and plot modules
import numpy as np
import meshio
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sfepy.discrete import FieldVariable

# Femtoscope functions
from femtoscope import RESULT_DIR, MESH_DIR, GEO_DIR
from femtoscope.inout.meshfactory import adjust_boundary_nodes
from femtoscope.inout.meshfactory import generate_mesh_from_geo
from femtoscope.inout.postprocess import ResultsPostProcessor as RPP
from femtoscope.physics.physical_problems import LinearSolver
from femtoscope.physics.physical_problems import Poisson


# Main class creation
class ForceOnTwoHollowCylinders:
    """
    Represents the physical system of two nested hollow cylinders. Each
    cylinder has internal and external radius, lenght and mass density.

    The first cylinder is placed inside the second, and is smaller than it.

    They are oriented such as the axis of the outer cylinder is considered the
    z-axis in cartesian framework.

    The movement of the inner cylinder can be following z-axis or x-axis, and
    the rotation is supposed around y-axis.

    There are two frameworks, called R1 and R2 respectively: one has the inner
    mass symmetrical with respect to its z-axis, the other has the outer mass
    symmetrical with respect to is own z-axis.

    In R1, only the inner mass has a non-null density, and in R2 only the outer
    mass has a non-null density. The null-density test mass is also meshed in
    both frameworks.
    """


    def __init__(self, problem_name: str, R_int_1: float, R_ext_1: float,
                 h_1: float, rho_1: float, R_int_2: float, R_ext_2: float,
                 h_2: float, rho_2: float, minSize: float, maxSize: float,
                 Z_1=0, R_1=0, rho_domain=0., tag_cyl_1=300, tag_cyl_2=301,
                 tag_domain_int=302, tag_domain_ext=303, tag_boundary_int=200,
                 tag_boundary_ext=201, FEM_ORDER=1, SOLVER='ScipyDirect',
                 VERBOSE=1):

        # Directly initializing the parameters given in the constructor
        self.problem_name = problem_name
        self.R_int_1 = R_int_1
        self.R_ext_1 = R_ext_1
        self.h_1 = h_1
        self.rho_1 = rho_1
        self.R_int_2 = R_int_2
        self.R_ext_2 = R_ext_2
        self.h_2 = h_2
        self.rho_2 = rho_2
        self.minSize = minSize
        self.maxSize = maxSize
        self.Z_1 = Z_1
        self.R_1 = R_1
        self.rho_domain = rho_domain
        self.tag_cyl_1 = tag_cyl_1
        self.tag_cyl_2 = tag_cyl_2
        self.tag_domain_int = tag_domain_int
        self.tag_domain_ext = tag_domain_ext
        self.tag_boundary_int = tag_boundary_int
        self.tag_boundary_ext = tag_boundary_ext
        self.FEM_ORDER = FEM_ORDER
        self.SOLVER = SOLVER
        self.VERBOSE = VERBOSE

        # Initializing the derived parameters
        x_Omega = np.max((self.R_ext_2 + self.R_1, self.R_ext_2 - self.R_1))
        z_Omega = np.max((np.abs((self.h_2 / 2) + self.Z_1), np.abs((self.h_2 / 2) - self.Z_1)))
        self.R_Omega = np.sqrt(x_Omega**2 + z_Omega**2) * 2  # <-- Equivalent to "R_cut"
        self.Ngamma = int(np.pi * self.R_Omega / self.maxSize)  # <-- Number of nodes on the boundary curve

        # Defining verbosity of the script
        if VERBOSE == 1:  # --> first party and Sfepy verbosity
            self.SFEPY_VERBOSE = False
            self.VERBOSE = True
        elif VERBOSE == 2:  # --> only first party (me!) verbosity
            self.SFEPY_VERBOSE = True
            self.VERBOSE = True
        else:  # --> no verbosity at all
            self.SFEPY_VERBOSE = False
            self.VERBOSE = False


    # Geometry verifications to ensure the geometrycal coherence of the system
    def _geometry_verifications(self):
        """
        In order to ensure a proper geometrical file, some basic geometrical
        verifications (such as the masses are not entering each others) are
        conducted.

        Notes
        -----
        This function is private and should only be accessed by this specific
        class, in this specific context, as another system may not use thesame
        specifications.

        """
        raise NotImplementedError("To be implemented later, when the script works.")


    # Generation of the geo files and the meshes
    def mesh_generation(self, SHOW_MESH=False):
        """
        Generates the geometrical files that describes the problem and the
        respective mesh files in vtk format.

        Parameters
        ----------
        SHOW_MESH: bool, optional.
            Triggers peeking of the meshes as soon as they are completed.
            Default is False.

        Returns
        -------
        mesh_R1_int : str
            The path of the .vtk file of the internal mesh in the R1 framework.
        mesh_R1_ext : TYPE
            The path of the .vtk file of the external mesh of the R1 framework.
        mesh_R2_int : str
            The path of the .vtk file of the internal mesh of the R2 framework.
        mesh_R2_ext : TYPE
            The path of the .vtk file of the external mesh of the R2 framework.

        Notes
        -----
        The directory of the file is the same as the one contained in the
        GEO_DIR of the Femtoscope module.

        """

        # Telling information about the mesh to the user if they want to
        if self.VERBOSE:
            print("\n=== MESH CHARACTERISTICS ===")
            print(" - R_int_1: {}".format(self.R_int_1))
            print(" - R_ext_1: {}".format(self.R_ext_1))
            print(" - h_1: {}".format(self.h_1))
            print(" - Vertical displacement: {}".format(self.Z_1))
            print(" - Radial displacement: {}".format(self.R_1))
            print(" - R_int_2: {}".format(self.R_int_2))
            print(" - R_ext_2: {}".format(self.R_ext_2))
            print(" - h_2: {}".format(self.h_2))
            print(" - R_Omega: {}".format(self.R_Omega))
            print(" - Distance between cylinders: {} m".format(self.R_int_2 - self.R_ext_1))
            print(" - Mesh's minimum size: {}".format(self.minSize))
            print(" - Mesh's maximum size: {}".format(self.maxSize))


        if self.VERBOSE:
            print("\n=== INTERNAL MESHES GENERATION ===")

        # Compiling the parameters for the R1 and R2 inner frameworks
        # NOTE: R_X and Z_X are only used in respective geo files!
        param_dict_int = {'R_int_1': self.R_int_1,
                          'R_ext_1': self.R_ext_1,
                          'h_1': self.h_1,
                          'R_1': self.R_1,  # This is used in the R1 geo file
                          'Z_1': self.Z_1,  # This too
                          'R_int_2': self.R_int_2,
                          'R_ext_2': self.R_ext_2,
                          'h_2': self.h_2,
                          'R_2': -self.R_1,  # And this one only in R2
                          'Z_2': -self.Z_1,  # Same for this
                          'R_Omega': self.R_Omega,
                          'Ngamma': self.Ngamma,
                          'minSize': self.minSize,
                          'maxSize': self.maxSize}

        if self.VERBOSE:
            print("Generating R1...", end="")

        # Generating inner R1 vtk mesh file
        self.mesh_R1_int = generate_mesh_from_geo(self.problem_name+'_2D_R1_int',
                                                  show_mesh=SHOW_MESH,
                                                  param_dict=param_dict_int,
                                                  VERBOSE=self.SFEPY_VERBOSE)

        if self.VERBOSE:
            print("\rR1 mesh is ready.\nGenerating R2...", end="")

        # Generating inner R2 vtk mesh file
        self.mesh_R2_int = generate_mesh_from_geo(self.problem_name+'_2D_R2_int',
                                                  show_mesh=SHOW_MESH,
                                                  param_dict=param_dict_int,
                                                  VERBOSE=self.SFEPY_VERBOSE)

        if self.VERBOSE:
            print("\rR2 mesh is ready.")


        if self.VERBOSE:
            print("\n=== EXTERNAL MESHES GENERATION ===")

        # Compiling the parameters for R1 and R2 outer frameworks
        # NOTE: R1 and R2 have similar external meshes, and parameters are shared
        param_dict_ext = {'R_Omega': self.R_Omega,
                          'Ngamma': self.Ngamma,
                          'minSize': self.minSize,
                          'maxSize': self.maxSize}

        if self.VERBOSE:
            print("Generating R1...", end="")

        self.mesh_R1_ext = generate_mesh_from_geo(self.problem_name+'_2D_R1_ext',
                                                  show_mesh=SHOW_MESH,
                                                  param_dict=param_dict_ext,
                                                  verbose=self.SFEPY_VERBOSE)

        if self.VERBOSE:
            print("\rR1 mesh is ready.\nGenerating R2...", end="")

        self.mesh_R2_ext = generate_mesh_from_geo(self.problem_name+'_2D_R2_ext',
                                                  show_mesh=SHOW_MESH,
                                                  param_dict=param_dict_ext,
                                                  verbose=self.SFEPY_VERBOSE)

        if self.VERBOSE:
            print("\rR2 mesh is ready.")

        # Adjusting the nodes between internal and external meshes
        # NOTE: this should be unnecessary, but since it's not so long we'll keep it
        adjust_boundary_nodes(self.mesh_R1_int, self.mesh_R1_ext,
                              self.tag_boundary_int, self.tag_boundary_ext)
        adjust_boundary_nodes(self.mesh_R2_int, self.mesh_R2_ext,
                              self.tag_boundary_int, self.tag_boundary_ext)


    # Method to solve the system for newtonian gravity
    def get_newton_potential(self, return_results=True):
        """
        Calculates the newtonian gravitational potential on both R1 and R2
        frameworks. Higher FEM_ORDER values improve the precision of the
        final results, but also increase the solving time.

        Parameters
        ----------
        return_results: bool, optional
            Triggers the return funtion. If False, only stores the results
            files in the RESULT_DIR directory. Default is True.

        Returns
        -------
        result_pp: array of RPP
            Triggered by return_results, it's an array that contains the result
            files ready to be post-processed.
        """

        if self.VERBOSE:
            print("\n=== NEWTONIAN FORCE COMPUTATION ===")
            print(" - FEM complexity: {}° order".format(self.FEM_ORDER))
            print(" - Solver name: {}".format(self.SOLVER))
            print("")


        # Constant intervening in the newtonian potential's computation
        G = 6.6743e-11  # [N*m²/kg²]
        ALPHA = 4 * np.pi * G


        if self.VERBOSE:
            print("=== SECOND FRAMEWORK COMPUTATION===")
            print("Setting R2 solver...", end="")

        poisson_R2 = Poisson({'alpha': ALPHA},
                             dim=2,
                             Rc=self.R_Omega,
                             coorsys="cylindrical")

        part_args_dict_R2_int = {'dim': 2,
                                 'name': 'wf_int',
                                 'pre_mesh': self.mesh_R2_int,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}

        density_dict_R2 = {('subomega', self.tag_cyl_1): self.rho_1,
                           ('subomega', self.tag_cyl_2): self.rho_2,
                           ('subomega', self.tag_domain_int): self.rho_domain}

        poisson_R2.set_wf_int(part_args_dict_R2_int, density_dict_R2)

        part_args_dict_R2_ext = {'dim': 2,
                                 'name': 'wf_ext',
                                 'pre_mesh': self.mesh_R2_ext,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma,
                                 'pre_ebc_dict': {('vertex', 0): self.rho_domain}}

        poisson_R2.set_wf_ext(part_args_dict_R2_ext, density=None)

        solver_R2 = LinearSolver(poisson_R2.wf_dict, ls_class=self.SOLVER,
                                 region_key_int=('facet', self.tag_boundary_int),
                                 region_key_ext=('facet', self.tag_boundary_ext))


        if self.VERBOSE:
            print("\rSolving R2...        ", end="")

        solver_R2.solve()


        try:
            solver_R2.save_results(self.problem_name + '_2D_R2_newton')
            print("\rDone.               \nResult saved.\n")

        except FileExistsError:
            result_path = RESULT_DIR / str(self.problem_name + '_2D_R2_newton')
            rmtree(result_path)
            solver_R2.save_results(self.problem_name + '_2D_R2_newton')
            print("\rDone.               \nResult saved.\n")


        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_results:
            return [None, RPP.from_files(self.problem_name + '_2D_R2_newton')]



    # Post-processing phase
    def invisible_postprocessing(self, result_pp_R1: RPP, result_pp_R2: RPP):
        """
        Trying to generate something good from what I've done.

        Parameters
        ----------
        result_pp_R1 : RPP
            More or less obvious. Don't put R2 in here.
        result_pp_R2 : RPP
            Same. They are not optional.

        Returns
        -------
        None.

        """

        if self.VERBOSE:
            print("\n=== INVISIBLE POST-PROCESSING ===")


        """
        Performing a regular post-process as if we were dealing with a
        parallel-only system. This should be used as a test to see if the data
        we are dealing with are good.
        """

        # Using the R2 framework to obtain the vertical gradient
        coors_R2 = result_pp_R2.coors_int
        wf_R2 = result_pp_R2.wf_int
        param = FieldVariable('param', 'parameter', wf_R2.field,
                              primary_var_name=wf_R2.get_unknown_name('cst'))
        param.set_data(coors_R2[:, 0] * result_pp_R2.sol_int)

        expression_IS1_R2 = "ev_grad.{}.subomega300(param)".format(wf_R2.integral.order)
        grad_Phi_IS1_R2 = wf_R2.pb_cst.evaluate(expression_IS1_R2,
                                                var_dict={'param': param})
        F_C1 = -grad_Phi_IS1_R2 * self.rho_1 * 2 * np.pi



        print("Force on vertical (z) axis on IS1 - R2:", F_C1[1])
        print("Stop.")


#%%

''' === VARIABLES DEFINITION === '''
# All the values are given in standard units
# First cylinder
R_int_1 = 15.4e-3
R_ext_1 = 19.7e-3
h_1 = 43.37e-3
rho_1 = 19972
Z_1 = -1e-5
R_1 = 0

# Second cylinder
R_int_2 = 30.4e-3
R_ext_2 = 34.6975e-3
h_2 = 79.83e-3
rho_2 = 4420
#rho_2 = 18000

# Miscellaneous
FILENAME = 'translation_rotation_hollow_cylinders'
VERBOSE = 1
FEM_ORDER = 2

# Physical parameters
SOLVER = 'ScipyDirect'

# Mesh size
minSize = 0.0005
maxSize = 0.005
''' === END OF VARIABLES DECLARATION === '''

# Creating the problem
FO2PHC = ForceOnTwoHollowCylinders(problem_name=FILENAME, R_int_1=R_int_1,
                                   R_ext_1=R_ext_1, rho_1=rho_1, h_1=h_1,
                                   R_int_2=R_int_2,
                                   R_ext_2=R_ext_2, h_2=h_2, rho_2=rho_2,
                                   Z_1=Z_1, R_1=R_1,
                                   minSize=minSize, maxSize=maxSize,
                                   VERBOSE=VERBOSE,
                                   FEM_ORDER=FEM_ORDER, SOLVER=SOLVER)

FO2PHC.mesh_generation(SHOW_MESH=False)

print("\n===NEWTONIAN GRAVITY ===")
results_pp_newton = FO2PHC.get_newton_potential(return_results=True)

FO2PHC.invisible_postprocessing(results_pp_newton[0], results_pp_newton[1])