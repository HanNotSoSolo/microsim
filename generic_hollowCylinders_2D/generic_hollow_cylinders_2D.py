#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 15:36:53 2025

@author: mdellava
"""

# Core modules, useful to manipulate files in the computer
from shutil import rmtree  # to remove the existing result

# Maths and plot functions
import numpy as np
from sfepy.discrete import FieldVariable
import gc

# Femtoscope functions
from femtoscope import RESULT_DIR, GEO_DIR
from femtoscope.inout.meshfactory import adjust_boundary_nodes
from femtoscope.inout.meshfactory import generate_mesh_from_geo
from femtoscope.inout.postprocess import ResultsPostProcessor as RPP
from femtoscope.physics.physical_problems import Poisson, Yukawa, LinearSolver
from cylinder_gravity.gravity.solids.cylinder import cylinder
from cylinder_gravity.gravity.solids.solidsPair import cylinderPair



class ForceOnTwoParallelCylinders:

    def __init__(self, problemName: str, R_int_1: float, R_ext_1: float,
                 rho_1: float, h_1: float, R_int_2: float, R_ext_2: float,
                 h_2: float, rho_2: float, minSize: float,
                 maxSize: float, Z_1=0, R_1=0, dim=2, rho_domain=0, rho_q_1=0,
                 rho_q_2=0, rho_q_domain=0, tag_cyl_1=300, tag_cyl_2=301,
                 tag_domain_int=302, tag_domain_ext=303, tag_boundary_int=200,
                 tag_boundary_ext=201, coorsys='cylindrical', FEM_ORDER=1,
                 SOLVER='ScipyDirect', VERBOSE=1):
        self.problemName = problemName
        self.R_int_1 = R_int_1
        self.R_ext_1 = R_ext_1
        self.rho_1 = rho_1
        self.h_1 = h_1
        self.R_int_2 = R_int_2
        self.R_ext_2 = R_ext_2
        self.h_2 = h_2
        self.rho_2 = rho_2
        self.Z_1 = Z_1
        self.R_1 = R_1
        self.minSize = minSize
        self.maxSize = maxSize
        self.dim = dim
        self.rho_domain = rho_domain
        self.rho_q_1 = rho_q_1
        self.rho_q_2 = rho_q_2
        self.rho_q_domain = rho_q_domain
        self.tag_cyl_1 = tag_cyl_1
        self.tag_cyl_2 = tag_cyl_2
        self.tag_domain_int = tag_domain_int
        self.tag_domain_ext = tag_domain_ext
        self.tag_boundary_int = tag_boundary_int
        self.tag_boundary_ext = tag_boundary_ext
        self.coorsys = coorsys
        self.FEM_ORDER = FEM_ORDER
        self.SOLVER = SOLVER
        self.VERBOSE = VERBOSE

        # Derived parameters that are a function of the previous ones
        self.R_Omega = np.sqrt(R_ext_2**2 + np.max((h_1, h_2))**2/4) * 2
        self.Ngamma = int(np.pi * self.R_Omega / self.maxSize)

        # The verbosity instructions for the script
        if VERBOSE == 0:  # --> no verbosity at all
            self.SFEPY_VERBOSE = False
            self.VERBOSE = False
        elif VERBOSE == 1:  # --> only first party (me!) verbosity
            self.SFEPY_VERBOSE = False
            self.VERBOSE = True
        elif VERBOSE == 2:  # --> first party and Sfepy verbosity
            self.SFEPY_VERBOSE = True
            self.VERBOSE = True


    def GEOMETRY_VERIFICATIONS(self):
        assert self.R_int_1 < self.R_ext_1, "Warning, first cylinder geometry problem."
        assert self.R_int_2 < self.R_ext_2, "Warning, second cylinder geometry problem."
        assert self.R_ext_1 + self.R_1 < self.R_int_2, "Warning, cylinders' geometry is inconsistent."
        assert self.R_int_1 + self.R_1 > 0, "Warning, first cylinder's radial displacement is inconsistent."
        assert self.h_1 > 0 and self.h_2 > 0, "Warning, invalid height encountered."
        assert self.R_int_1 > 0 and self.R_int_2 > 0, "Warning, invalid radius encountered."

        if self.VERBOSE:
            print("Geometry verifications: OK.\n")


    def mesh_generation(self, SHOW_MESH=False):
        """
        Generation of the geometrical files that will compose the system. Note
        that the .geo files must have the same name as the problem. Opposely to
        the parallel hollow cylinders' case, since we don't have a cylindrical
        symmetry anymore, we create two meshes for two different frameworks:
        one representing the "POV" of the first cylinder, and the other one the
        second cylinder's POV.

        Parameters
        ----------
        SHOW_MESH : bool, optional
            If True, a window will appear to show the mesh and the program will
            pause until the window is closed. The default is False.

        Returns
        -------
        mesh_R1_int : str
            The path of the .vtk file of the internal mesh in the R1 framework.
            The directory of the file is the same as the one contained in the
            GEO_DIR of the Femtoscope module.
        mesh_R1_ext : TYPE
            The path of the .vtk file of the external mesh of the R1 framework.
        mesh_R2_int : str
            The path of the .vtk file of the internal mesh of the R2 framework.
        mesh_R2_ext : TYPE
            The path of the .vtk file of the external mesh of the R2 framework.

        """

        # Telling information about the mesh to the user if they want to
        if self.VERBOSE:
            print("=== MESH CHARACTERISTICS ===")
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
            print(" - Mesh type: {}D".format(self.dim))
            print(" - Coordinates system: {} framework\n".format(self.coorsys))

        if self.VERBOSE:
            print("=== INNER MESHES GENERATION ===")

        param_dict_R1_int = {'R_int_1': self.R_int_1,
                             'R_ext_1': self.R_ext_1,
                             'h_1': self.h_1,
                             'Z_2': self.Z_1,  # to understand this parameter, please refer to the .geo file
                             'R_2': self.R_1,  # (respective displacements are equal and opposite)
                             'R_int_2': self.R_int_2,
                             'R_ext_2': self.R_ext_2,
                             'h_2': self.h_2,
                             'R_Omega': self.R_Omega,
                             'minSize': self.minSize,
                             'maxSize': self.maxSize,
                             'Ngamma': self.Ngamma}
        mesh_R1_int = generate_mesh_from_geo(self.problemName + '_R1_int',
                                             show_mesh=SHOW_MESH,
                                             param_dict=param_dict_R1_int,
                                             verbose=self.SFEPY_VERBOSE)

        param_dict_R2_int = {'R_int_1': self.R_int_1,
                             'R_ext_1': self.R_ext_1,
                             'h_1': self.h_1,
                             'Z_1': self.Z_1,
                             'R_1': self.R_1,
                             'R_int_2': self.R_int_2,
                             'R_ext_2': self.R_ext_2,
                             'h_2': self.h_2,
                             'R_Omega': self.R_Omega,
                             'minSize': self.minSize,
                             'maxSize': self.maxSize,
                             'Ngamma': self.Ngamma}
        mesh_R2_int = generate_mesh_from_geo(self.problemName + '_R2_int',
                                             show_mesh=SHOW_MESH,
                                             param_dict=param_dict_R2_int,
                                             verbose=self.SFEPY_VERBOSE)
        if self.VERBOSE:
            print("OK.\n")

        if self.VERBOSE:
            print("=== OUTER MESH GENERATION ===")

        # Generating external mesh
        param_dict_ext = {'R_Omega': self.R_Omega,
                          'minSize': self.minSize,
                          'maxSize': self.maxSize,
                          'Ngamma': self.Ngamma}
        mesh_R1_ext = generate_mesh_from_geo(self.problemName + '_R2_ext',
                                          show_mesh=SHOW_MESH,
                                          param_dict=param_dict_ext,
                                          verbose=self.SFEPY_VERBOSE)

        # External meshes are the same for both frameworks
        mesh_R2_ext = mesh_R1_ext

        if self.VERBOSE:
            print("OK.\n")

        # This function is not essential for 2D, but it's a redundancy that both boundary curves correspond
        adjust_boundary_nodes(mesh_R1_int, mesh_R1_ext, self.tag_boundary_int,
                              self.tag_boundary_ext)
        adjust_boundary_nodes(mesh_R2_int, mesh_R2_ext, self.tag_boundary_int,
                              self.tag_boundary_ext)


        return mesh_R1_int, mesh_R1_ext, mesh_R2_int, mesh_R2_ext


    def get_newton_potential(self, mesh_R1_int: str, mesh_R1_ext: str,
                             mesh_R2_int: str, mesh_R2_ext: str,
                             return_result=True):
        """
        Calculates the newtonian gravitational potential on the .vtk file. The
        potential is calculated on every node of the mesh, and interpolated
        between the nodes with a FEM_ORDER-degree polynomial, depending on
        what has been declared during the class creation.

        Parameters
        ----------
        mesh_int : str
            The path of the .vtk internal mesh file.
        mesh_ext : str
            The path of the .vtk external mesh file.
        return_result : bool, optional
            If True, the function not only solves the system for every node and
            saves it in the data/results/ directory, but also returns the
            results file in a variable. The default is True.

        Returns
        -------
        result_pp : ResultsPostProcessor
            When asked, returns the results file that will be used for
            post-processing operations. Triggered by return_result parameter.

        """

        if self.VERBOSE:
            print("=== NEWTONIAN FORCE COMPUTATION ===")
            print(" - FEM complexity: {}° order".format(self.FEM_ORDER))
            print(" - Solver name: {}".format(self.SOLVER))
            print("")

        # Creating the constants that will caracterize the Newton's version of
        # the Poisson's problem
        G = 6.6743e-11
        ALPHA = 4 * np.pi * G

        # Starting computation of the first framework
        if self.VERBOSE:
            print("== FIRST FRAMEWORK COMPUTATION ==")

        poisson_R1 = Poisson({'alpha': ALPHA}, dim=self.dim, Rc=self.R_Omega,
                             coorsys=self.coorsys)

        partial_args_dict_R1_int = {'dim': self.dim,
                                    'name': 'wf_int',
                                    'pre_mesh': mesh_R1_int,
                                    'fem_order': self.FEM_ORDER,
                                    'Ngamma': self.Ngamma}

        # Attributing a density only to the first cylinder
        poisson_R1.set_wf_int(partial_args_dict_R1_int,
                              {('subomega', self.tag_cyl_1): self.rho_1,
                               ('subomega', self.tag_cyl_2): self.rho_domain,
                               ('subomega', self.tag_domain_int): self.rho_domain})

        partial_args_dict_R1_ext = {'dim': self.dim,
                                    'name': 'wf_ext',
                                    'pre_mesh': mesh_R1_ext,
                                    'fem_order': self.FEM_ORDER,
                                    'Ngamma': self.Ngamma}
        partial_args_dict_R1_ext['pre_ebc_dict'] = {('vertex', 0): self.rho_domain}
        poisson_R1.set_wf_ext(partial_args_dict_R1_ext, density=None)

        poisson_R1_solver = LinearSolver(poisson_R1.wf_dict, ls_class=self.SOLVER,
                                         region_key_int=('facet',
                                                         self.tag_boundary_int),
                                         region_key_ext=('facet',
                                                         self.tag_boundary_ext))
        poisson_R1_solver.solve()

        ''' This part is here to ensure the result is correctly saved'''

        try:
            poisson_R1_solver.save_results(self.problemName + '_R1_newton')
            print("First framework's result saved.\n")

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problemName + '_R1_newton')
            rmtree(resultPath)
            poisson_R1_solver.save_results(self.problemName + '_R1_newton')
            print("First framework's result saved.\n")

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        # Starting computation of the second framework
        if self.VERBOSE:
            print("== SECOND FRAMEWORK COMPUTATION ==")

        poisson_R2 = Poisson({'alpha': ALPHA}, dim=self.dim, Rc=self.R_Omega,
                             coorsys=self.coorsys)

        partial_args_dict_R2_int = {'dim': self.dim,
                                    'name': 'wf_int',
                                    'pre_mesh': mesh_R2_int,
                                    'fem_order': self.FEM_ORDER,
                                    'Ngamma': self.Ngamma}

        # Attributing a density only to the second cylinder
        poisson_R2.set_wf_int(partial_args_dict_R2_int,
                              {('subomega', self.tag_cyl_1): self.rho_domain,
                               ('subomega', self.tag_cyl_2): self.rho_2,
                               ('subomega', self.tag_domain_int): self.rho_domain})

        partial_args_dict_R2_ext = {'dim': self.dim,
                                    'name': 'wf_ext',
                                    'pre_mesh': mesh_R2_ext,
                                    'fem_order': self.FEM_ORDER,
                                    'Ngamma': self.Ngamma}
        partial_args_dict_R2_ext['pre_ebc_dict'] = {('vertex', 0): self.rho_domain}
        poisson_R2.set_wf_ext(partial_args_dict_R2_ext, density=None)

        poisson_R2_solver = LinearSolver(poisson_R2.wf_dict, ls_class=self.SOLVER,
                                         region_key_int=('facet',
                                                         self.tag_boundary_int),
                                         region_key_ext=('facet',
                                                         self.tag_boundary_ext))
        poisson_R2_solver.solve()

        ''' This part is here to ensure the result is correctly saved'''

        try:
            poisson_R1_solver.save_results(self.problemName + '_R2_newton')
            print("Second framework's result saved.\n")

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problemName + '_R2_newton')
            rmtree(resultPath)
            poisson_R2_solver.save_results(self.problemName + '_R2_newton')
            print("Second framework's result saved.\n")

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_result:
            return np.array([RPP.from_files(self.problemName + '_R1_newton'), RPP.from_files(self.problemName + '_R2_newton')])


    def get_electrostatic_potential(self, mesh_int, mesh_ext, return_result=True):
        """
        Calculates the electrostatic potential on the .vtk file. The potential
        is calculated on every node of the mesh, and interpolated between the
        nodes with a FEM_ORDER-degree polynomial, depending on what has been
        declared during the class creation.

        Parameters
        ----------
        mesh_int : str
            The path of the .vtk internal mesh file.
        mesh_ext : str
            The path of the .vtk external mesh file.
        return_result : bool, optional
            If True, the function not only solves the system for every node and
            saves it in the data/results/ directory, but also returns the
            results file in a variable. The default is True.

        Returns
        -------
        result_pp : ResultsPostProcessor
            When asked, returns the results file that will be used for
            post-processing operations. Triggered by return_result parameter.

        """

        raise NotImplementedError("Sorry, this is not ready yet!")

        if self.VERBOSE:
            print("=== ELECTROSTATIC FORCE COMPUTATION ===")
            print(" - FEM complexity: {}° order".format(self.FEM_ORDER))
            print(" - Solver name: {}".format(self.SOLVER))
            print("")


        # Creating the constants that will caracterize the Coulomb's version of
        # the Poisson's problem
        epsilon_void = 8.8541878128e-12  # [F / m]
        ALPHA = epsilon_void**(-1)

        poisson = Poisson({'alpha': ALPHA}, dim=self.dim, Rc=self.R_Omega,
                          coorsys=self.coorsys)

        partial_args_dict_int = {'dim': self.dim,
                                 'name': 'wf_int',
                                 'pre_mesh': mesh_int,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}
        poisson.set_wf_int(partial_args_dict_int,
                           {('subomega', self.tag_cyl_1): self.rho_q_1,
                            ('subomega', self.tag_cyl_2): self.rho_q_2,
                            ('subomega', self.tag_domain_int): self.rho_q_domain})

        partial_args_dict_ext = {'dim': self.dim,
                                 'name': 'wf_ext',
                                 'pre_mesh': mesh_ext,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}
        partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 0): self.rho_q_domain}
        poisson.set_wf_ext(partial_args_dict_ext, density=None)

        poisson_solver = LinearSolver(poisson.wf_dict, ls_class=self.SOLVER,
                                      region_key_int=('facet',
                                                      self.tag_boundary_int),
                                      region_key_ext=('facet',
                                                      self.tag_boundary_ext))

        poisson_solver.solve()

        ''' This part is here to ensure the result is correctly saved'''

        try:
            poisson_solver.save_results(self.problemName + '_electrostatic')
            print('Result saved.\n')

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problemName + '_electrostatic')
            rmtree(resultPath)
            poisson_solver.save_results(self.problemName + '_electrostatic')
            print('Result saved.\n')

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_result:
            return RPP.from_files(self.problemName + '_electrostatic')


    def get_yukawa_potential(self, mesh_int, mesh_ext, alpha: float,
                             lmbda: float, L_0=1, rho_0=1, return_result=True):
        """
        Solves the Klein-Gordon equation applied to the internal and external
        meshes according to the (alpha, lambda) couple given by the user.
        The equation that this method solves is as follows:
            ƔΔU = U + ρ
        !!! NOT SURE ANYMORE ABOUT GAMMA AND L_0, REFER TO -SOMEONE- FOR HELP

        Parameters
        ----------
        mesh_int : str
            The address of the internal domain's mesh. It's usually located in
            the femtoscope.GEO_DIR directory.
        mesh_ext : str
            The address of the external domain's mesh. It's usually located in
            the femtoscope.GEO_DIR directory.
        alpha : float
            The α parameter, representing the intensity of the Yukawa
            contribution to the gravitational interaction.
        lmbda : float
            The Ⲗ parameter, representing the range of the Yukawa interaction.
        L_0 : float, optional
            The caracteristic lenght of the problem, useful simplify the
            computation for the computer. The default is 1.
        rho_0 : float, optional
            The caracteristic mass density of the problem, useful to simplify
            the computation for the computer. The default is 1.
        return_result : bool, optional
            If True, the method not only solves the problem and stores the
            result but also returns a file that will be used for
            post-processing. The default is False.

        Returns
        -------
        result_pp : ResultsPostProcessor
            When asked, returns the results file that will be used for
            post-processing operations. Triggered by return_result parameter.

        """

        raise NotImplementedError("Sorry, this is not ready yet!")

        if self.VERBOSE:
            print("=== YUKAWA FORCE COMPUTATION ===")
            print(" - FEM complexity: {}° order".format(self.FEM_ORDER))
            print(" - Solver name: {}".format(self.SOLVER))
            print(" - scale factor α: {}".format(alpha))
            print(" - range factor Ⲗ: {}".format(lmbda))
            print(" - characteristic lenght: {}m".format(L_0))
            print(" - characteristic density: {} [kg·m^-3]".format(rho_0))
            print("")

        # Creating the gamma parameter
        gamma = lmbda**2 / L_0**2

        # Defining the problem and its metadata
        yukawa = Yukawa({'gamma': gamma}, dim=self.dim, Rc=self.R_Omega,
                        coorsys=self.coorsys)

        # Internal domain caracteristics
        partial_args_dict_int = {'dim': self.dim,
                                 'name': 'wf_int',
                                 'pre_mesh': mesh_int,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}
        yukawa.set_wf_int(partial_args_dict_int,
                           {('subomega', self.tag_cyl_1): self.rho_1 / rho_0,
                            ('subomega', self.tag_cyl_2): self.rho_2 / rho_0,
                            ('subomega', self.tag_domain_int): self.rho_domain / rho_0})

        # External domain caracteristics
        partial_args_dict_ext = {'dim': self.dim,
                                 'name': 'wf_ext',
                                 'pre_mesh': mesh_ext,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}
        partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 0): self.rho_domain / rho_0}
        yukawa.set_wf_ext(partial_args_dict_ext, density=None)


        # Setting up the solver
        yukawa_solver = LinearSolver(yukawa.wf_dict, ls_class='ScipyDirect',
                                     region_key_int=('facet',
                                                     self.tag_boundary_int),
                                     region_key_ext=('facet',
                                                     self.tag_boundary_ext))
        yukawa_solver.solve()

        ''' This part is here to ensure the result is correctly saved'''

        try:
            yukawa_solver.save_results(self.problemName + '_yukawa')
            print('Result saved.\n')

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problemName + '_yukawa')
            rmtree(resultPath)
            yukawa_solver.save_results(self.problemName + '_yukawa')
            print('Result saved.\n')

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_result:
            return RPP.from_files(self.problemName + '_yukawa')


    def postprocess_force(self, postprocess_file_R1, postprocess_file_R2,
                          alpha=0, lmbda=1, rho_0=1, getNewton=False,
                          getCoulomb=False, getYukawa=False):
        """
        Computes the gravitational and electrostatic force between the two
        parallel hollow cylinders. For gravity, can compute Newton's or
        Yukawa's potential.

        Parameters
        ----------
        postprocess_file_R1 : ResultsPostProcessor
            The results file representing cylinder 1's framework.
        postprocess_file_R2 : ResultsPostProcessor
            The results file representing cylinder 2's framework.
        alpha : float, optional
            The scale factor used to compute the Yukawa potential, is useless
            for other computations. The default is 0.
        lmbda : float, optional
            The range factor used to compute the Yukawa potential, is useless
            for the other computations. The default is 1.
        rho_o : float, optional
            The caracteristic density of the problem, it's used for the
            rescaling of the Yukawa potential's result. NOTE that this value
            must be equal to the one declared for the resolution function!. The
            default is 1.
        getNewton : bool, optional
            When True, activates the post-processing of newton's gravitational
            potential. The default is False.
        getCoulomb : bool, optional
            When True, activates the post-processing of the electrostatic
            potential. The default is False.
        getYukawa : bool, optional
            When True, activates the post-processing of yukawa's gravitational
            potential. The default is False.
        compare : bool, optional
            When True, performs an additional analytical calculation that will
            compute the relative error. The analytical calculation is the one
            that has been used for MICROSCOPE mission (see [1]). The default is
            True.

        Returns
        -------
        grad_phi_cyl_1[1]: float
            The vertical force applied on the first hollow cylinder.
        grad_phi_cyl_2[1]: float
            The vertical force applied on the second hollow cylinder.
        F_ana: float
            The analytically calculated vertical force.
        epsilon: float
            The relative error of the method compared with analytical method.

        References
        ----------
        [1] Bergé, J.
            (2023).
            MICROSCOPE’s view at gravitation.
            Reports on Progress in Physics, 86(6), 066901.

        """

        #  Asserting that we have exactly one force to calculate
        A = getNewton and not getCoulomb and not getYukawa
        B = not getNewton and getCoulomb and not getYukawa
        C = not getNewton and not getCoulomb and getYukawa
        assert A or B or C, "Please select exactly one force for post-process!"


        if getNewton or getYukawa:
            rho_1 = self.rho_1
            rho_2 = self.rho_2
            rho_domain = self.rho_domain
            k = 6.6743e-11
        elif getCoulomb:
            rho_1 = self.rho_q_1
            rho_2 = self.rho_q_2
            rho_domain = self.rho_q_domain
            k = (4 * np.pi * 8.8541878128e-12)**-1


        if getYukawa:
            raise NotImplementedError("Sorry, coming soon! <3")

            # Informing the user about the characteristics of the K-G problem
            if self.VERBOSE:
                print(" === Linear Klein-Gordon problem ===")
                print(" - scale factor α: {}".format(alpha))
                print(" - range factor Ⲗ: {}".format(lmbda))
                print(" - rho_1: {} [kg·m^-3]".format(self.rho_1))
                print(" - rho_2: {} [kg·m^-3]".format(self.rho_2))
                print(" - rho_domain: {} [kg·m^-3]".format(self.rho_domain))
                print("")

            # Computing nondimensioning term U_0
            U_0 = 4 * np.pi * k * lmbda**2 * alpha * rho_0

            # Extractign the weak form
            wf_int_yukawa = postprocess_file.wf_int

            # Setting the FieldVariable for the potential integration
            param_yukawa = FieldVariable('param', 'parameter',
                                         wf_int_yukawa.field,
                                         primary_var_name=wf_int_yukawa.get_unknown_name('cst'))

            # Setting the weight of the potential's value according to r
            param_yukawa.set_data(coors_int[:, 0] * postprocess_file.sol_int)

            # Integrating the potential on the whole surface - Yukawa S1
            expression_yukawa_C1 = "ev_grad.{}.subomega300(param)".format(wf_int_yukawa.integral.order)
            grad_yukawa_C1 = -wf_int_yukawa.pb_cst.evaluate(expression_yukawa_C1,
                                                            var_dict={'param': param_yukawa}) * rho_1 * 2 * np.pi * U_0

            # Integrating the potential on the whole surface - Yukawa S2
            expression_yukawa_C2 = "ev_grad.{}.subomega301(param)".format(wf_int_yukawa.integral.order)
            grad_yukawa_C2 = -wf_int_yukawa.pb_cst.evaluate(expression_yukawa_C2,
                                                            var_dict={'param': param_yukawa}) * rho_2 * 2 * np.pi * U_0

            # Communicating the results to the user
            print("Force on cylinder 1:", str(grad_yukawa_C1[1]), "N")

            print("Force on cylinder 2:", str(grad_yukawa_C2[1]), "N")

            # Analytical calculation and comparison
            if self.VERBOSE:
                print("Computing analytical force...", end="")

            # Creating the two cylinders to compute analitically
            ana_c1 = cylinder(radius=self.R_ext_1, height=self.h_1,
                              density=self.rho_1, innerRadius=self.R_int_1)
            ana_c2 = cylinder(radius=self.R_ext_2, height=self.h_2,
                              density=self.rho_2, innerRadius=self.R_int_2)

            # Creating the interaction between the two cylinders
            c1_c2 = cylinderPair(ana_c1, ana_c2)

            # Computing the force on the Z axis of the cylinder
            ana_Fz_1 = c1_c2.cmpAnaFz([self.R_1, 0, self.Z_1], [0, 0, 0],
                                      _dir='2->1', yukawa=True, lmbda=lmbda,
                                      alpha=alpha)
            ana_Fz_2 = c1_c2.cmpAnaFz([self.R_1, 0, self.Z_1], [0, 0, 0],
                                      _dir='1->2', yukawa=True, lmbda=lmbda,
                                      alpha=alpha)

            # Computing relative error
            epsilon_1 = (grad_yukawa_C1[1] - ana_Fz_1) / grad_yukawa_C1[1]
            epsilon_2 = (grad_yukawa_C2[1] - ana_Fz_2) / grad_yukawa_C2[1]

            # Communicating the results
            print("\r                             ", end="")  # <-- this is here to make everything prettier
            print("\rAnalytical force:", str(ana_Fz_1), "N")
            print("Relative error:", str(epsilon_1))

            return grad_yukawa_C1[1], grad_yukawa_C2[1], ana_Fz_1, ana_Fz_2, epsilon_1, epsilon_2

        else:

            # Informing the user about the characteristics of the Poisson problem
            if self.VERBOSE:
                if getNewton:
                    print(" === Poisson problem - Newtonian ===")
                    print(" - rho_1: {} [kg·m^-3]".format(rho_1))
                    print(" - rho_2: {} [kg·m^-3]".format(rho_2))
                    print(" - rho_domain: {} [kg·m^-3]".format(rho_domain))
                    print("")
                elif getCoulomb:
                    raise NotImplementedError("Sorry, coming soon! <3")
                    print(" === Poisson problem - Electrostatic ===")
                    print(" - rho_1: {} [C·m^-3]".format(rho_1))
                    print(" - rho_2: {} [C·m^-3]".format(rho_2))
                    print(" - rho_domain: {} [C·m^-3]".format(rho_domain))
                    print("")





            # Taking the potential's gradient for the two cylinders in R1
            wf_R1 = postprocess_file_R1.wf_int
            param_R1 = FieldVariable('param', 'parameter', wf_R1.field,
                                  primary_var_name=wf_R1.get_unknown_name('cst'))
            pseudo_coors_int_R2 = postprocess_file_R1.coors_int + np.array([self.R_1, self.Z_1])
            image_R2 = postprocess_file_R2.evaluate_at(pseudo_coors_int_R2,
                                                       mode='val')
            param_R1.set_data((postprocess_file_R1.coors_int[:, 0]* postprocess_file_R1.sol_int) + (pseudo_coors_int_R2[:, 0] * image_R2))

            expression_cylinder_1_R1 = "ev_grad.{}.subomega300(param)".format(wf_R1.integral.order)
            expression_cylinder_2_R1 = "ev_grad.{}.subomega301(param)".format(wf_R1.integral.order)

            force_cylinder_1_R1 = -wf_R1.pb_cst.evaluate(expression_cylinder_1_R1,
                                                         var_dict={'param': param_R1}) * rho_1 * 2 * np.pi
            force_cylinder_2_R1 = -wf_R1.pb_cst.evaluate(expression_cylinder_2_R1,
                                                         var_dict={'param': param_R1}) * rho_2 * 2 * np.pi

            # Taking the potential's gradient for the two cylinders in R2
            wf_R2 = postprocess_file_R2.wf_int
            param_R2 = FieldVariable('param', 'parameter', wf_R2.field,
                                  primary_var_name=wf_R2.get_unknown_name('cst'))
            param_R2.set_data(postprocess_file_R2.coors_int[:, 0] * postprocess_file_R2.sol_int)

            expression_cylinder_1_R2 = "ev_grad.{}.subomega300(param_R2)".format(wf_R2.integral.order)
            expression_cylinder_2_R2 = "ev_grad.{}.subomega301(param_R2)".format(wf_R2.integral.order)

            force_cylinder_1_R2 = -wf_R2.pb_cst.evaluate(expression_cylinder_1_R2,
                                                         var_dict={'param': param_R2}) * rho_1 * 2 * np.pi
            force_cylinder_2_R2 = -wf_R2.pb_cst.evaluate(expression_cylinder_2_R2,
                                                         var_dict={'param': param_R2}) * rho_2 * 2 * np.pi




            pass

            print("Force on cylinder 1:", str(grad_phi_cylinder_1[1]), "N")
            print("Force on cylinder 2:", str(grad_phi_cylinder_2[1]), "N")

            # Analytical calculation and comparison
            if self.VERBOSE:
                print("Computing analytical force...", end="")

            # Creating the two cylinders to compute analitically
            ana_c1 = cylinder(radius=self.R_ext_1, height=self.h_1,
                              density=self.rho_1, innerRadius=self.R_int_1)
            ana_c2 = cylinder(radius=self.R_ext_2, height=self.h_2,
                              density=self.rho_2, innerRadius=self.R_int_2)

            # Creating the interaction between the two cylinders
            c1_c2 = cylinderPair(ana_c1, ana_c2)

            # Computing the force on the Z axis of the cylinder
            ana_Fz_1 = c1_c2.cmpAnaFz([self.R_1, 0, self.Z_1], [0, 0, 0],
                                    _dir='2->1')
            ana_Fz_2 = c1_c2.cmpAnaFz([self.R_1, 0, self.Z_1], [0, 0, 0],
                                    _dir='1->2')

            # Computing relative error
            epsilon_1 = np.abs((grad_phi_cylinder_1[1] - ana_Fz_1) / grad_phi_cylinder_1[1])
            epsilon_2 = np.abs((grad_phi_cylinder_2[1] - ana_Fz_2) / grad_phi_cylinder_2[1])

            # Communicating the results
            print("\r                             ", end="")  # <-- this is here to make everything prettier
            print("\rAnalytical force:", str(ana_Fz_1), "N")
            print("Relative error:", str(epsilon_1))

            # Returning the results
            return grad_phi_cylinder_1[1], grad_phi_cylinder_2[1], ana_Fz_1, ana_Fz_2, epsilon_1, epsilon_2


#%% Testing the class

#@profile

def test():
    """
    Feel free to use this as an example of use case.

    Returns
    -------
    None.

    """

    ''' === VARIABLES DEFINITION === '''
    # All the values are given in standard units
    # First cylinder
    R_int_1 = 15.4e-3
    R_ext_1 = 19.7e-3
    h_1 = 43.37e-3
    rho_1 = 19972
    rho_q_1 = 0
    Z_1 = -1e-5
    R_1 = -1e-5

    # Second cylinder
    R_int_2 = 30.4e-3
    R_ext_2 = 34.6975e-3
    h_2 = 79.83e-3
    rho_2 = 4420
    rho_q_2 = 0

    # Miscellaneous
    FILENAME = 'generic_hollow_cylinders_2D'
    VERBOSE = 0
    FEM_ORDER = 1

    # Physical parameters
    DIM = 2
    COORSYS = 'cylindrical'
    SOLVER = 'ScipyDirect'

    # Mesh size
    minSize = 0.0005
    maxSize = 0.001
    ''' === END OF VARIABLES DECLARATION === '''

    # Creating the problem
    FO2PHC = ForceOnTwoParallelCylinders(problemName=FILENAME, R_int_1=R_int_1,
                                         R_ext_1=R_ext_1, rho_1=rho_1, h_1=h_1,
                                         rho_q_1=rho_q_1, R_int_2=R_int_2,
                                         R_ext_2=R_ext_2, h_2=h_2, rho_2=rho_2,
                                         rho_q_2=rho_q_2, Z_1=Z_1, R_1=R_1,
                                         minSize=minSize, maxSize=maxSize,
                                         dim=DIM, VERBOSE=VERBOSE,
                                         FEM_ORDER=FEM_ORDER, SOLVER=SOLVER,
                                         coorsys=COORSYS)

    # Conducting critical verifications
    FO2PHC.GEOMETRY_VERIFICATIONS()

    # Creating the meshes
    mesh_R1_int, mesh_R1_ext, mesh_R2_int, mesh_R2_ext= FO2PHC.mesh_generation()

    print("\n === NEWTONIAN GRAVITY ===")
    result_pp_newton = FO2PHC.get_newton_potential(mesh_R1_int, mesh_R1_ext,
                                                   mesh_R2_int, mesh_R2_ext)
    F_N_1, F_N_2, F_N_ana, _, epsilon_N, _ = FO2PHC.postprocess_force(result_pp_newton[0],
                                                                      result_pp_newton[1],
                                                                      getNewton=True)

    # print("\n === ELECTROSTATIC FORCE ===")
    # result_pp_elec = FO2PHC.get_electrostatic_potential(mesh_int, mesh_ext)
    # F_E_1, F_E_2 = FO2PHC.postprocess_force(result_pp_elec, getCoulomb=True)

    # print("\n === YUKAWA GRAVITY ===")
    # alpha = 1  # scale factor compared to Newton potential
    # lmbda = 10000  # range factor (large lmbda leads to Newton potential)
    # rho_0 = 10e4
    # L_0 = 1  # FIXME L_0 not equal to 1 makes the problem explode!

    # result_pp_yukawa = FO2PHC.get_yukawa_potential(mesh_int, mesh_ext,
    #                                                 alpha=alpha, lmbda=lmbda,
    #                                                 rho_0=rho_0, L_0=L_0)
    # F_Y_1, F_Y_2, F_Y_ana, _, epsilon_Y, _ = FO2PHC.postprocess_force(result_pp_yukawa, alpha=alpha,
    #                                         lmbda=lmbda, rho_0=rho_0,
    #                                         getYukawa=True)


test()