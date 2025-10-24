#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:06:44 2025

@author: mdellava
"""

# Core modules, useful to manipulate files in the computer
from shutil import rmtree  # to remove the existing result
import gc  # garbage collector, to manage memory

# Maths and plot functions
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from sfepy.discrete import FieldVariable

# Femtoscope functions
from femtoscope import RESULT_DIR, GEO_DIR
from femtoscope.inout.meshfactory import adjust_boundary_nodes
from femtoscope.inout.meshfactory import generate_mesh_from_geo
from femtoscope.inout.postprocess import ResultsPostProcessor as RPP
from femtoscope.physics.physical_problems import Poisson, Yukawa, LinearSolver
from femtoscope.misc.analytical import potential_sphere


class ForceOnTwoSpheres:
    """
    Creates a two-spheres problem, and the functions can analyze the
    interactions between the two.

    Parameters
    ----------
    problemName : str
        The name of the geometric file of the problem, without any appendix or
        extension.
    R_1 : float
        The radius of the first (upper) sphere. [m]
    rho_1 : float
        The mass density of the first sphere. [kg/m^3]
    R_2 : float
        The radius of the second (bottom) sphere. [m]
    rho_2 : float
        The mass density of the second sphere. [kg/m^3]
    d : float
        The distance between the centre of the two spheres.
    minSize : float
        The size of the elements near to the interface areas of the spheres.
    maxSize : float
        The size of the elements far from the interface areas, or where the
        potential doesn't change so much.
    dim : int, optional
        The dimensions of the problem. THIS VALUE SHOULD NOT BE CHANGED. The
        default is 2.
    rho_domain : float, optional
        The mass density of the domain. The default is 0.
    rho_q_1 : float, optional
        The charge density of the first sphere. The default is 0. [C/m^3]
    rho_q_2 : float, optional
        The charge density of the second sphere. The default is 0. [C/m^3]
    rho_q_domain : float, optional
        The charge density of the domain. The default is 0. [C/m^3]
    tag_sphere_1 : int, optional
        The tag value of the first sphere's physical group. It's addressed by
        the user, and appears in the Gmsh geometry file. The default is 300.
    tag_sphere_2 : int, optional
        The tag value of the second sphere's physical group. It's addressed by
        the user, and appears in the Gmsh geometry file. The default is 301.
    tag_domain_int : int, optional
        The tag value of the internal domain's physical group. It's addressed
        by the user, and appears in the Gmsh geometry file. The default is 302.
    tag_domain_ext : int, optional
        The tag value of the external domain's physical group. It's addressed
        by the user, and appears in the Gmsh geometry file. The default is 303.
    tag_boundary_int : int, optional
        The tag value of the internal boundary curve's physical group. It's
        addressed by the user, an appears in the Gmsh geometry file. The
        default is 200.
    tag_boundary_ext : int, optional
        The tag value of the external boundary curve's physical group. It's
        addressed by the user, an appears in the Gmsh geometry file. The
        default is 201.
    coorsys : str, optional
        The coorsinates system used to describe the problem. Can be
        'cartesian', 'cylindrical or 'spherical', but is usually defined
        BEFORE the creation of the rest of the problem and should not be
        changed.The default is 'cylindrical'.
    FEM_ORDER : int, optional
        The order of the FEM calculation. Minimum is 1, and upper values
        implies more solving time and memory, but there are no upper limits.
        The default is 2.
    SOLVER : str, optional
        Used to solve the equation system. Can be chosen between 'ScipyDirect',
        'ScipyIterative', 'ScipySuperLU', 'ScipyUmfpack' and 'MUMPS'.
    VERBOSE : int, optional
        Sets the verbosity level. 0: no message will be displayed; 1: only the
        first-party functions will have verbosity; 2: all functions will have
        verbosity, when it's available. Default is 1.

    Methods
    -------
    GEOMETRY_VERIFICATIONS()
        Perform some basic geometry verifications related to the .geo file. Not
        mandatory, but if the computation fails, it's the first thing to check.
    mesh_generation()
        Gives the meshes of the internal and external domains based on the
        caracteristics given while creating the class.
    get_newton_force(mesh_int, mesh_ext)
        Solves the Poisson problem for the newtonian gravitation. It creates a
        results file with the same name of the problem, ending with '_ΔΦ'.
    get_electrostatic_force(mesh_int, mesh_ext)
        Solves the Poisson problem for the Coulomb electrostatic interaction.
        It creates a results file with the same name of the problem, ending
        with '_Δϕ'.
    postprocess_force(results_file)
        Computes the results file to get the force resulting on the potential
        computed during the previous phase.

    """

    def __init__(self, problemName, R_1: float, rho_1: float, R_2: float,
                 rho_2: float, d: float, minSize: float, maxSize: float,
                 dim: int, rho_domain=0, rho_q_1=0, rho_q_2=0, rho_q_domain=0,
                 tag_sphere_1=300, tag_sphere_2=301, tag_domain_int=302,
                 tag_domain_ext=303, tag_boundary_int=200,
                 tag_boundary_ext=201, coorsys='cylindrical', FEM_ORDER=1,
                 SOLVER='ScipyDirect', VERBOSE=1):
        # Directly initialized parameters
        self.problemName = problemName  # the geometrical file name without the extention
        self.R_1 = R_1  # All measures are expressed in S.I. units
        self.R_2 = R_2  # These, as an example, are meters
        self.d = d  # [m]
        self.minSize = minSize  # The size of the mesh next to the interfaces
        self.maxSize = maxSize  # The size of the mesh far from the spheres
        self.dim = dim  # The dimensions of the problem (=2 for sure)
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.rho_domain = rho_domain
        self.rho_q_1 = rho_q_1
        self.rho_q_2 = rho_q_2
        self.rho_q_domain = rho_q_domain
        self.tag_sphere_1 = tag_sphere_1
        self.tag_sphere_2 = tag_sphere_2
        self.tag_domain_int = tag_domain_int
        self.tag_domain_ext = tag_domain_ext
        self.tag_boundary_int = tag_boundary_int
        self.tag_boundary_ext = tag_boundary_ext
        self.coorsys = coorsys
        self.FEM_ORDER = FEM_ORDER
        self.SOLVER = SOLVER

        # Derived parameters that are a function of the previous ones
        self.R_Omega = (self.d + max((R_1, R_2))) * 2
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
        '''
        Due to the configuration of the geometry file (which is entirely
        arbitrary and can be changed by future users), it's possible that one
        of the two spheres crosses the middle of the scheme, which would result
        in a failure of the geometry definitions. To avoid this, we can simply
        put a verification phase that asserts that this isn't the case.

        '''
        assert self.R_1 < (self.d/2), "Sphere 1 (up one) is too big, please reduce its size or increase the distance."
        assert self.R_2 < (self.d/2), "Sphere 2 (down one) is too big, please reduce its size or increase the distance."
        assert (self.R_1 + self.R_2) < self.d, "Spheres intersect, please increase the distance between them."

        if self.VERBOSE:
            print("Geometry verifications: OK.\n")


    def mesh_generation(self, SHOW_MESH=False):
        """
        Generation of the geometrical files that will compose the system. Note
        that the .geo files must have the same name as the problem.

        Parameters
        ----------
        SHOW_MESH : bool, optional
            If True, a window will appear to show the mesh and the program will
            pause until the window is closed. The default is False.

        Returns
        -------
        mesh_int : str
            The path of the .vtk file of the internal mesh. The directory is
            the same given by the variable MESH_DIR from femtoscope module.
        mesh_ext : TYPE
            The path of the .vtk file of the external mesh. The directory is
            the same given by the variable MESH_DIR from femtoscope module.

        """

        # Telling information about the mesh to the user if they want to
        if self.VERBOSE:
            print("=== MESH CHARACTERISTICS ===")
            print(" - R_1: {}".format(self.R_1))
            print(" - R_2: {}".format(self.R_2))
            print(" - R_Omega: {}".format(self.R_Omega))
            print(" - Distance centre-to-centre: {} m".format(self.d))
            print(" - Mesh's minimum size: {}".format(self.minSize))
            print(" - Mesh's maximum size: {}".format(self.maxSize))
            print(" - Mesh type: {}D".format(self.dim))
            print(" - Coordinates system: {} framework\n".format(self.coorsys))

        if self.VERBOSE:
            print("=== INNER MESH GENERATION ===")

        param_dict_int = {'R_1': self.R_1,
                          'R_2': self.R_2,
                          'd': self.d,
                          'R_Omega': self.R_Omega,
                          'minSize': self.minSize,
                          'maxSize': self.maxSize,
                          'Ngamma': self.Ngamma}
        mesh_int = generate_mesh_from_geo( self.problemName + '_int',
                                          show_mesh=SHOW_MESH,
                                          param_dict=param_dict_int,
                                          verbose=self.SFEPY_VERBOSE)
        if self.VERBOSE:
            print("OK.\n")

        if self.VERBOSE:
            print("=== OUTER MESH GENERATION ===")

        param_dict_ext = {'R_Omega': self.R_Omega,
                          'minSize': self.minSize,
                          'maxSize': self.maxSize,
                          'Ngamma': self.Ngamma}
        mesh_ext = generate_mesh_from_geo(self.problemName + '_ext',
                                          show_mesh=SHOW_MESH,
                                          param_dict=param_dict_ext,
                                          verbose=self.SFEPY_VERBOSE)
        if self.VERBOSE:
            print("OK.\n")

        adjust_boundary_nodes(mesh_int, mesh_ext, self.tag_boundary_int,
                              self.tag_boundary_ext)

        return mesh_int, mesh_ext


    def get_newton_potential(self, mesh_int, mesh_ext, return_result=True):
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

        poisson = Poisson({'alpha': ALPHA}, dim=self.dim, Rc=self.R_Omega,
                          coorsys=self.coorsys)

        partial_args_dict_int = {'dim': self.dim,
                                 'name': 'wf_int',
                                 'pre_mesh': mesh_int,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}
        poisson.set_wf_int(partial_args_dict_int,
                           {('subomega', self.tag_sphere_1): self.rho_1,
                            ('subomega', self.tag_sphere_2): self.rho_2,
                            ('subomega', self.tag_domain_int): self.rho_domain})

        partial_args_dict_ext = {'dim': self.dim,
                                 'name': 'wf_ext',
                                 'pre_mesh': mesh_ext,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}
        partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 0): self.rho_domain}
        poisson.set_wf_ext(partial_args_dict_ext, density=None)

        poisson_solver = LinearSolver(poisson.wf_dict, ls_class=self.SOLVER,
                                      region_key_int=('facet',
                                                      self.tag_boundary_int),
                                      region_key_ext=('facet',
                                                      self.tag_boundary_ext))
        poisson_solver.solve()

        ''' This part is here to ensure the result is correctly saved'''

        try:
            poisson_solver.save_results(self.problemName + '_newton')
            print('Result saved.\n')

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problemName + '_newton')
            rmtree(resultPath)
            poisson_solver.save_results(self.problemName + '_newton')
            print('Result saved.\n')

        if return_result:
            return RPP.from_files(self.problemName + '_newton')


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
                           {('subomega', self.tag_sphere_1): self.rho_q_1,
                            ('subomega', self.tag_sphere_2): self.rho_q_2,
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

        if return_result:
            return RPP.from_files(self.problemName + '_electrostatic')


    def get_yukawa_potential(self, mesh_int, mesh_ext, alpha: float, lmbda: float,
                         L_0=1, rho_0=1, return_result=True):
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
                           {('subomega', self.tag_sphere_1): self.rho_1 / rho_0,
                            ('subomega', self.tag_sphere_2): self.rho_2 / rho_0,
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

        if return_result:
            return RPP.from_files(self.problemName + '_yukawa')


    def Phi(self, x: float):
        """
        A core function for the Yukawa gravitational potential.

        Parameters
        ----------
        x : float
            The argument of the function. Should represent (R/Ⲗ).

        Returns
        -------
        Phi : float
              Φ(x) = 3(x cosh x − sinh x)/x 3 normally, or an approximate value
              if needed.

        References
        ----------
        [1] Adelberger, E. G., Heckel, B. R., & Nelson, A. E.
            (2003).
            Tests of the gravitational inverse-square law.
            arXiv preprint hep-ph/0307284.

        """

        # Case disjunction: using Taylor's decomposition
        if x > 100 :
            phi = 3 * mp.exp(x) / (2 * x**2)

        elif x < 0.001:
            phi = 1

        else:
            phi = 3 * ( x * mp.cosh(x) - mp.sinh(x)) / x**3

        return phi


    def postprocess_force(self, postprocess_file, alpha=0, lmbda=1, rho_0=1,
                          getNewton=False, getCoulomb=False, getYukawa=False):
        """
        Computes the gravitational and electrostatic force between the two
        spheres. For gravity, can compute Newton's or Yukawa's potential.

        Parameters
        ----------
        postprocess_file : ResultsPostProcessor
            The results file that the user wants to compute.
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

        Returns
        -------
        grad_phi_sphere_1[1]: float
            The vertical force applied on the first sphere.
        grad_phi_sphere_2[1]: float
            The vertical force applied on the second sphere.
        epsilon : float
            The precision of the force on the first sphere with respect to the
            analytical calculation.

        References
        ----------
        [1] Adelberger, E. G., Heckel, B. R., & Nelson, A. E.
            (2003).
            Tests of the gravitational inverse-square law.
            arXiv preprint hep-ph/0307284.

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

        # Extracting coordinates from results file
        # NOTE: nodes' coordinates should match since it's the same mesh file
        coors_int = postprocess_file.coors_int

        if getYukawa:

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
            expression_yukawa_S1 = "ev_grad.{}.subomega300(param)".format(wf_int_yukawa.integral.order)
            grad_yukawa_S1 = -wf_int_yukawa.pb_cst.evaluate(expression_yukawa_S1,
                                                            var_dict={'param': param_yukawa}) * rho_1 * 2 * np.pi * U_0

            # Integrating the potential on the whole surface - Yukawa S2
            expression_yukawa_S2 = "ev_grad.{}.subomega301(param)".format(wf_int_yukawa.integral.order)
            grad_yukawa_S2 = -wf_int_yukawa.pb_cst.evaluate(expression_yukawa_S2,
                                                            var_dict={'param': param_yukawa}) * rho_2 * 2 * np.pi * U_0

            # Communicating the results to the user
            print("Force on sphere 1:", str(grad_yukawa_S1[1]), "N")

            print("Force on sphere 2:", str(grad_yukawa_S2[1]), "N")

            # Calculating analytical Yukawa's potential (see [1])
            m_1 = (4 / 3) * np.pi * self.R_1**3 * self.rho_1
            m_2 = (4 / 3) * np.pi * self.R_2**3 * self.rho_2
            F_ana = -(alpha * k * m_1 * m_2 * self.Phi(self.R_1/lmbda) *
                      self.Phi(self.R_2/lmbda) * (1 + (self.d / lmbda)) *
                      mp.exp(-self.d/lmbda) / self.d**2)

            print("Analytically calculated force :", str(F_ana) + " N")
            epsilon = np.abs((grad_yukawa_S1[1] - F_ana) / grad_yukawa_S1[1])
            print("Precision on vertical force: ", epsilon)
            print("\n\n")

            return grad_yukawa_S1[1], grad_yukawa_S2[1], F_ana, epsilon

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
                    print(" === Poisson problem - Electrostatic ===")
                    print(" - rho_1: {} [C·m^-3]".format(rho_1))
                    print(" - rho_2: {} [C·m^-3]".format(rho_2))
                    print(" - rho_domain: {} [C·m^-3]".format(rho_domain))
                    print("")

            # Formatting the request for Sfepy
            wf_int = postprocess_file.wf_int
            param = FieldVariable('param', 'parameter', wf_int.field,
                                  primary_var_name=wf_int.get_unknown_name('cst'))
            param.set_data(coors_int[:, 0] * postprocess_file.sol_int)

            expression_sphere_1 = "ev_grad.{}.subomega300(param)".format(wf_int.integral.order)
            grad_phi_sphere_1 = -wf_int.pb_cst.evaluate(expression_sphere_1,
                                                        var_dict={'param': param}) * rho_1 * 2 * np.pi

            expression_sphere_2 = "ev_grad.{}.subomega301(param)".format(wf_int.integral.order)
            grad_phi_sphere_2 = -wf_int.pb_cst.evaluate(expression_sphere_2,
                                                        var_dict={'param': param}) * rho_2 * 2 * np.pi

            print("Force on sphere 1:", str(grad_phi_sphere_1[1]), "N")
            print("Force on sphere 2:", str(grad_phi_sphere_2[1]), "N")

            # Force precision verification
            F_ana = - k * (((4/3) * np.pi)**2 * self.R_1**3 *
                           rho_1 * self.R_2**3 * rho_2) / self.d**2

            epsilon = np.abs((F_ana - grad_phi_sphere_1[1]) / F_ana)

            print("Precision on vertical force: ", epsilon)
            print("\n\n")

            return grad_phi_sphere_1[1], grad_phi_sphere_2[1], epsilon


    def newton_residual_map(self, postprocess_file, save_figure=True):
        """
        Shows the relative error of the computation for every point of the
        mesh in the internal domain. The result is compared to the analytical
        calculation, which is supposed to be exact.

        Parameters
        ----------
        postprocess_file : RPP
            The results .vtk file generated after the solving stage.
        save_figure : Bool, optional
            If True, the figure is not only shown (if possible), but also saved
            in the same folder as the script. The default is True.

        Returns
        -------
        None.

        """


        # Getting evaluated potential
        V_fem = postprocess_file.sol_int

        # Getting internal domain's coordinates
        coors_int = postprocess_file.coors_int

        # Calculating the distance of the points from the centre of the spheres
        r_1 = np.sqrt((-coors_int[:, 0])**2 + (self.d/2 - coors_int[:, 1])**2)
        r_2 = np.sqrt((-coors_int[:, 0])**2 + (self.d/2 + coors_int[:, 1])**2)

        # Evaluating the potential of the first sphere in every node
        V_S1_ana = potential_sphere(r_1, self.R_1, 6.6743e-11, rho=self.rho_1)
        V_S2_ana = potential_sphere(r_2, self.R_2, 6.6743e-11, rho=self.rho_2)

        # Applying the superposition theorem
        V_ana = V_S1_ana + V_S2_ana

        # Calculating the residual
        res = np.abs((V_ana - V_fem) / V_ana)

        # Plotting the result
        plt.tricontourf(coors_int[:, 0], coors_int[:, 1], res)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='relative error')#, format='%4.2e')

        if save_figure:
            plt.savefig('Newton_gravity_potential_errormap.png')




# %% Testing the class

def test():
    """
    Feel free to use this as an example of use case.

    Returns
    -------
    None.

    """
    ''' === VARIABLES DEFINITION ==='''
    # All the values are given in standard units
    # First sphere
    R_1 = 2  # [m]
    rho_1 = 19972  # [kg/m^3]
    rho_q_1 = 19e-8  # [C/m^3]

    # Second sphere
    R_2 = 2
    rho_2 = 4278
    rho_q_2 = -19e-8

    # Domain sphere
    d = 4.001  # The distance between the centres of the spheres

    # Miscellaneous
    FILENAME = 'spheres_2D'
    VERBOSE = 1

    # Mesh meta-parameters

    # Physical parameters
    DIM = 2
    COORSYS = 'cylindrical'
    SOLVER = 'ScipyDirect'

    # Mesh size
    minSize = 0.05
    maxSize = 0.5
    ''' === END OF VARIABLES DECLARATION ==='''


    # Creating the meshes that will be used by the solvers
    FO2S = ForceOnTwoSpheres(problemName=FILENAME, R_1=R_1, rho_1=rho_1,
                             R_2=R_2, rho_2=rho_2, d=d, minSize=minSize,
                             maxSize=maxSize, dim=DIM, rho_q_1=rho_q_1,
                             rho_q_2=rho_q_2, coorsys=COORSYS, SOLVER=SOLVER,
                             VERBOSE=VERBOSE)

    # Verifying some critical issues of the mesh
    FO2S.GEOMETRY_VERIFICATIONS()

    # Actually creating the meshes
    mesh_int, mesh_ext = FO2S.mesh_generation()

    print("\n === NEWTONIAN GRAVITY ===")
    result_pp_newton = FO2S.get_newton_potential(mesh_int, mesh_ext)
    F_N, _, epsilon_N = FO2S.postprocess_force(result_pp_newton,
                                                getNewton=True)
    FO2S.newton_residual_map(result_pp_newton, save_figure=False)

    print("\n === ELECTROSTATIC FORCE ===")
    result_pp_elec = FO2S.get_electrostatic_potential(mesh_int, mesh_ext)
    F_E, _, epsilon_E = FO2S.postprocess_force(result_pp_elec,
                                                getCoulomb=True)

    print("\n === YUKAWA GRAVITY ===")
    alpha = 5e-1
    lmbda = 0.01
    rho_0 = 10e4
    L_0 = 0.1  # FIXME L_0 not equal to 1 makes the problem explode!

    result_pp_yukawa= FO2S.get_yukawa_potential(mesh_int, mesh_ext,
                                                alpha=alpha, lmbda=lmbda,
                                                L_0=L_0, rho_0=rho_0)
    F_Y, _, F_ana_Y, epsilon_Y = FO2S.postprocess_force(postprocess_file=result_pp_yukawa,
                                                        alpha=alpha,
                                                        lmbda=lmbda,
                                                        rho_0=rho_0,
                                                        getYukawa=True)

    # Deleting the files that are still in memory that Python won't automatically delete
    # NOTE: without this line, everything works but memory usage increase for serial resolutions
    gc.collect()

#test()
