# Core modules, useful to manipulate files in the computer
from shutil import rmtree  # to remove the existing result

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


class ForceOnTwoParallelCylinders:

    def __init__(self, problemName, R_int_1: float, R_ext_1: float,
                 rho_1: float, h_1: float, R_int_2: float, R_ext_2: float,
                 h_2: float, rho_2: float, Z_1: float, minSize: float,
                 maxSize: float, dim=2, rho_domain=0, rho_q_1=0, rho_q_2=0,
                 rho_q_domain=0, tag_cyl_1=300, tag_cyl_2=301,
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
        self.R_Omega = np.sqrt(R_ext_2**2 + h_2**2/4) * 2
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
        assert self.R_ext_1 < self.R_int_2, "Warning, cylinders' geometry is inconsistent."
        assert self.h_1 > 0 and self.h_2 > 0, "Warning, invalid height encountered."

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
            print(" - R_int_1: {}".format(self.R_int_1))
            print(" - R_ext_1: {}".format(self.R_ext_1))
            print(" - R_int_2: {}".format(self.R_int_2))
            print(" - R_ext_2: {}".format(self.R_ext_2))
            print(" - R_Omega: {}".format(self.R_Omega))
            print(" - Distance between cylinders: {} m".format(self.R_int_2 - self.R_ext_1))
            print(" - Mesh's minimum size: {}".format(self.minSize))
            print(" - Mesh's maximum size: {}".format(self.maxSize))
            print(" - Mesh type: {}D".format(self.dim))
            print(" - Coordinates system: {} framework\n".format(self.coorsys))

        if self.VERBOSE:
            print("=== INNER MESH GENERATION ===")

        param_dict_int = {'R_int_1': self.R_int_1,
                          'R_ext_1': self.R_ext_1,
                          'R_int_2': self.R_int_2,
                          'R_ext_2': self.R_ext_2,
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