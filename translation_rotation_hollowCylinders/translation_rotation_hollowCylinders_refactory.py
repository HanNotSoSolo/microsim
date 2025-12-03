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

    def __init__(self, problem_name, R_int_1: float, R_ext_1: float,
                 rho_1: float, h_1: float, R_int_2: float, R_ext_2: float,
                 h_2: float, rho_2: float, minSize: float,
                 maxSize: float, Z_1=0, R_1=0, dim=2, rho_domain=0, rho_q_1=0,
                 rho_q_2=0, rho_q_domain=0, tag_cyl_1=300, tag_cyl_2=301,
                 tag_domain_int=302, tag_domain_ext=303, tag_boundary_int=200,
                 tag_boundary_ext=201, coorsys='cylindrical', FEM_ORDER=1,
                 SOLVER='ScipyDirect', VERBOSE=1):
        self.problem_name = problem_name
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
        self.R_Omega = np.sqrt(R_ext_2**2 + h_2**2) * 2
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
        """
        In order to ensure a proper geometrical file, some basic geometrical
        verifications (such as the masses are not entering each others) are
        conducted.

        Notes
        -----
        This function is private and should only be accessed by this specific
        class, in this specific context, as another system may not use the same
        specifications.

        """
        raise NotImplementedError("To be implemented later, when the script works.")



    def mesh_generation(self, mesh_name: str, SHOW_MESH=False):
        """
        Generates the geometrical files that describes the problem and the
        respective mesh files in vtk format.

        Parameters
        ----------
        mesh_path: str
            The name to the geometrical internal and external mesh. Usually
            it's problem_name, but can also have appendix like '_R1' or '_R2'.
        SHOW_MESH: bool, optional.
            Triggers peeking of the meshes as soon as they are completed.
            Default is False.

        Returns
        -------
        mesh_int : str
            The path of the .vtk file of the internal mesh.
        mesh_ext : TYPE
            The path of the .vtk file of the external mesh.

        Notes
        -----
        The directory of the file is the same as the one contained in the
        GEO_DIR of the Femtoscope module.

        """

        # Telling information about the mesh to the user if they want to
        if self.VERBOSE:
            print("=== MESH CHARACTERISTICS ===")
            print(" - R_int_1: {}".format(self.R_int_1))
            print(" - R_ext_1: {}".format(self.R_ext_1))
            print(" - h_1: {}".format(self.h_1))
            print(" - Vertical displacement: {}".format(self.Z_1))
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
            print("=== INNER MESH GENERATION ===")

        param_dict_int = {'R_int_1': self.R_int_1,
                          'R_ext_1': self.R_ext_1,
                          'h_1': self.h_1,
                          'R_1': self.R_1,
                          'Z_1': self.Z_1,
                          'R_int_2': self.R_int_2,
                          'R_ext_2': self.R_ext_2,
                          'R_2': -self.R_1,
                          'Z_2': -self.Z_1,
                          'h_2': self.h_2,
                          'R_Omega': self.R_Omega,
                          'minSize': self.minSize,
                          'maxSize': self.maxSize,
                          'Ngamma': self.Ngamma}
        mesh_int = generate_mesh_from_geo(mesh_name + '_int',
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
        mesh_ext = generate_mesh_from_geo(mesh_name + '_ext',
                                          show_mesh=SHOW_MESH,
                                          param_dict=param_dict_ext,
                                          verbose=self.SFEPY_VERBOSE)
        if self.VERBOSE:
            print("OK.\n")

        adjust_boundary_nodes(mesh_int, mesh_ext, self.tag_boundary_int,
                              self.tag_boundary_ext)

        return [mesh_int, mesh_ext]


    def get_newton_potential(self, res_name, mesh_int, mesh_ext,
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

        poisson = Poisson({'alpha': ALPHA}, dim=self.dim, Rc=self.R_Omega,
                          coorsys=self.coorsys)

        partial_args_dict_int = {'dim': self.dim,
                                 'name': 'wf_int',
                                 'pre_mesh': mesh_int,
                                 'fem_order': self.FEM_ORDER,
                                 'Ngamma': self.Ngamma}
        poisson.set_wf_int(partial_args_dict_int,
                           {('subomega', self.tag_cyl_1): self.rho_1,
                            ('subomega', self.tag_cyl_2): self.rho_2,
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
            poisson_solver.save_results(res_name + '_newton')
            print('Result saved.\n')

        except FileExistsError:
            resultPath = RESULT_DIR / str(res_name + '_newton')
            rmtree(resultPath)
            poisson_solver.save_results(res_name + '_newton')
            print('Result saved.\n')

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_result:
            return RPP.from_files(self.problem_name + '_newton')


    def get_electrostatic_potential(self, mesh_int, mesh_ext,
                                    return_result=True):
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
            poisson_solver.save_results(self.problem_name + '_electrostatic')
            print('Result saved.\n')

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problem_name + '_electrostatic')
            rmtree(resultPath)
            poisson_solver.save_results(self.problem_name + '_electrostatic')
            print('Result saved.\n')

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_result:
            return RPP.from_files(self.problem_name + '_electrostatic')


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
            yukawa_solver.save_results(self.problem_name + '_yukawa')
            print('Result saved.\n')

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problem_name + '_yukawa')
            rmtree(resultPath)
            yukawa_solver.save_results(self.problem_name + '_yukawa')
            print('Result saved.\n')

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_result:
            return RPP.from_files(self.problem_name + '_yukawa')


    def postprocess_force(self, postprocess_file, alpha=0, lmbda=1, rho_0=1,
                          getNewton=False, getCoulomb=False, getYukawa=False):
        """
        Computes the gravitational and electrostatic force between the two
        parallel hollow cylinders. For gravity, can compute Newton's or
        Yukawa's potential.

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

            expression_cylinder_1 = "ev_grad.{}.subomega300(param)".format(wf_int.integral.order)
            grad_phi_cylinder_1 = -wf_int.pb_cst.evaluate(expression_cylinder_1,
                                                          var_dict={'param': param}) * rho_1 * 2 * np.pi

            expression_cylinder_2 = "ev_grad.{}.subomega301(param)".format(wf_int.integral.order)
            grad_phi_cylinder_2 = -wf_int.pb_cst.evaluate(expression_cylinder_2,
                                                          var_dict={'param': param}) * rho_2 * 2 * np.pi

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
minSize = 0.0003
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

mesh_R1 = FO2PHC.mesh_generation(mesh_name=FILENAME+'_2D_R1', SHOW_MESH=False)
mesh_R2 = FO2PHC.mesh_generation(mesh_name=FILENAME+'_2D_R2', SHOW_MESH=False)

print("\n===NEWTONIAN GRAVITY ===")
results_pp_newton_R1 = FO2PHC.get_newton_potential(res_name=FILENAME+'_2D_R1',
                                                    mesh_int=mesh_R1[0],
                                                    mesh_ext=mesh_R1[1],
                                                    return_result=True)

results_pp_newton_R2 = FO2PHC.get_newton_potential(res_name=FILENAME+"_2D_R2",
                                                   mesh_int=mesh_R2[0],
                                                   mesh_ext=mesh_R2[1],
                                                   return_result=True)

results_R1 = FO2PHC.postprocess_force(results_pp_newton_R1, getNewton=True)

results_R2 = FO2PHC.postprocess_force(results_pp_newton_R2, getNewton=True)