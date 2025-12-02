#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 17:16:25 2025

@author: mdellava
"""
# Core modules, useful to manipulate files in the computer
from shutil import rmtree  # to remove the existing result

# Maths and plot functions
import numpy as np
from scipy.spatial import cKDTree
from sfepy.discrete import FieldVariable
import gc
import meshio
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Femtoscope functions
from femtoscope import RESULT_DIR, MESH_DIR, GEO_DIR
from femtoscope.inout.meshfactory import adjust_boundary_nodes
from femtoscope.inout.meshfactory import generate_mesh_from_geo
from femtoscope.inout.vtkfactory import create_connectivity_table
from femtoscope.inout.vtkfactory import create_structured_vtk
from femtoscope.inout.postprocess import ResultsPostProcessor as RPP
from femtoscope.physics.physical_problems import Poisson, Yukawa, LinearSolver
from femtoscope.core.pre_term import PreTerm
from femtoscope.core.weak_form import WeakForm
#from cylinder_gravity.gravity.solids.cylinder import cylinder
#from cylinder_gravity.gravity.solids.solidsPair import cylinderPair



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


    def mesh_generation(self, mesh_3D=False, SHOW_MESH=False):
        """
        Generation of the geometrical files that will compose the system. Note
        that the .geo files must have the same name as the problem. Opposely to
        the parallel hollow cylinders' case, since we don't have a cylindrical
        symmetry anymore, we create two meshes for two different frameworks:
        one representing the "POV" of the first cylinder, and the other one the
        second cylinder's POV.

        Parameters
        ----------
        mesh_3D : bool, optional
            If True, also meshes the 3D version of the system. Default is
            False.
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
        mesh_R1_int = generate_mesh_from_geo(self.problemName + '_2D_R1_int',
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
        mesh_R2_int = generate_mesh_from_geo(self.problemName + '_2D_R2_int',
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
        mesh_R1_ext = generate_mesh_from_geo(self.problemName + '_2D_R1_ext',
                                             show_mesh=SHOW_MESH,
                                             param_dict=param_dict_ext,
                                             verbose=self.SFEPY_VERBOSE)

        # External meshes are the same for both frameworks
        mesh_R2_ext = generate_mesh_from_geo(self.problemName + '_2D_R2_ext',
                                             show_mesh=SHOW_MESH,
                                             param_dict=param_dict_ext,
                                             verbose=self.SFEPY_VERBOSE)

        if self.VERBOSE:
            print("OK.\n")


        # Meshing the 3D version of the system (if activated)
        if mesh_3D:
            if self.VERBOSE:
                print("=== 3D MESH GENERATION ===")

            param_dict_3D = {"R_int_1": self.R_int_1,
                             "R_ext_1": self.R_ext_1,
                             "h_1": self.h_1,
                             "R_int_2": self.R_int_2,
                             "R_ext_2": self.R_ext_2,
                             "h_2": self.h_2,
                             "Z_2": self.Z_1,
                             "R_2": self.R_1,
                             "minSize": self.minSize
                             }
            mesh_3D = generate_mesh_from_geo(self.problemName + '_3D_R1',
                                             show_mesh=SHOW_MESH,
                                             param_dict=param_dict_3D,
                                             verbose=self.SFEPY_VERBOSE)

            if self.VERBOSE:
                print("OK.\n")

        else:
            mesh_3D = None

        # This function is not essential for 2D, but it's a redundancy that both boundary curves correspond
        adjust_boundary_nodes(mesh_R1_int, mesh_R1_ext, self.tag_boundary_int,
                              self.tag_boundary_ext)
        adjust_boundary_nodes(mesh_R2_int, mesh_R2_ext, self.tag_boundary_int,
                              self.tag_boundary_ext)


        return mesh_R1_int, mesh_R1_ext, mesh_R2_int, mesh_R2_ext, mesh_3D


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

        poisson_R1_solver = LinearSolver(poisson_R1.wf_dict,
                                         ls_class=self.SOLVER,
                                         region_key_int=('facet',
                                                         self.tag_boundary_int),
                                         region_key_ext=('facet',
                                                         self.tag_boundary_ext))
        poisson_R1_solver.solve()

        ''' This part is here to ensure the result is correctly saved'''

        try:
            poisson_R1_solver.save_results(self.problemName + '_2D_R1_newton')
            print("First framework's result saved.\n")

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problemName + '_2D_R1_newton')
            rmtree(resultPath)
            poisson_R1_solver.save_results(self.problemName + '_2D_R1_newton')
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
                              {('subomega', self.tag_cyl_1): self.rho_1,  # FIXME this should be rho_domain
                               ('subomega', self.tag_cyl_2): self.rho_2,
                               ('subomega', self.tag_domain_int): self.rho_domain})

        partial_args_dict_R2_ext = {'dim': self.dim,
                                    'name': 'wf_ext',
                                    'pre_mesh': mesh_R2_ext,
                                    'fem_order': self.FEM_ORDER,
                                    'Ngamma': self.Ngamma}
        partial_args_dict_R2_ext['pre_ebc_dict'] = {('vertex', 0): self.rho_domain}
        poisson_R2.set_wf_ext(partial_args_dict_R2_ext, density=None)

        poisson_R2_solver = LinearSolver(poisson_R2.wf_dict,
                                         ls_class=self.SOLVER,
                                         region_key_int=('facet',
                                                         self.tag_boundary_int),
                                         region_key_ext=('facet',
                                                         self.tag_boundary_ext))
        poisson_R2_solver.solve()

        ''' This part is here to ensure the result is correctly saved'''

        try:
            poisson_R2_solver.save_results(self.problemName + '_2D_R2_newton')
            print("Second framework's result saved.\n")

        except FileExistsError:
            resultPath = RESULT_DIR / str(self.problemName + '_2D_R2_newton')
            rmtree(resultPath)
            poisson_R2_solver.save_results(self.problemName + '_2D_R2_newton')
            print("Second framework's result saved.\n")

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if return_result:
            return np.array([RPP.from_files(self.problemName + '_2D_R1_newton'), RPP.from_files(self.problemName + '_2D_R2_newton')])


    def vtk_generation(self, result_pp_R1: RPP, result_pp_R2: RPP):

        # Creation of a new directory where to store the 3D result
        # # FIXME: this does not work when a folder is already present
        # dir_path = Path(RESULT_DIR / (self.problemName + "_3D"))
        # dir_path.mkdir()

        # Creation of the vtk file
        mesh_3D = meshio.read(Path(MESH_DIR / (self.problemName + "_3D_R1.vtk")))
        coors = mesh_3D.points
        #cells_1 = create_connectivity_table(coors)  # FIXME: remove later
        cells = mesh_3D.cells
        node_groups = mesh_3D.point_data
        coors_2D_R1 = np.column_stack((np.sqrt(coors[:, 0]**2 + coors[:, 1]**2), coors[:, 2]))
        coors_2D_R2 = np.column_stack((np.sqrt((coors[:, 0] - self.R_1)**2 + coors[:, 1]**2), coors[:, 2] - self.Z_1))
        sol_int = result_pp_R1.evaluate_at(coors_2D_R1) + result_pp_R2.evaluate_at(coors_2D_R2)
        vars_dict = {'sol_int': sol_int,
                     'physical_groups_nodes': node_groups['node_groups']}
        path_name = str(Path(RESULT_DIR / (self.problemName + '_3D') / (self.problemName + '_3D')))

        create_structured_vtk(coors, vars_dict, cells[0][1], path_name=path_name)





        # points_3D = mesh_3D.points
        # cells_3D = mesh_3D.cells

        # # Computing cylindrical coordinates
        # r_R1 = np.sqrt(points_3D[:, 0]**2 + points_3D[:, 1]**2)
        # z_R1 = points_3D[:, 2]
        # coors_R1 = np.column_stack((r_R1, z_R1))

        # # For every point of 3D mesh, computing the potential
        # sol_int = result_pp_R1.evaluate_at(coors_R1) + result_pp_R2.evaluate_at(coors_R1+[self.R_1, self.Z_1])
        # node_groups = np.zeros_like(sol_int)

        # # Creating a new mesh file that also has the potential's values
        # new_mesh_3D = meshio.Mesh(points=points_3D, cells=cells_3D,
        #                           point_data={"sol_int": sol_int,
        #                                       "node_groups": node_groups})

        # # Writing this mesh in a file
        # new_mesh_3D.write(new_mesh_filename, file_format="vtk")


    def pkl_generation(self):

        # # Creation of the pseudo-pre_term
        # pre_term = PreTerm(name='dw_laplace', region_key=('omega', -1),
        #                    tag='cst', prefactor=1.0, mat=None, mat_kwargs={})

        # # Arguments for the creation of the pkl file
        # args_dict = {'dim': self.dim,
        #              'pre_mesh': '/home/mdellava/Documenti/ONERA - Ph.D/femtoscope/translation_rotation_hollowCylinders/data/result/translation_rotation_hollow_cylinders_3D_R1/translation_rotation_hollow_cylinders_3D_R1',
        #              'name': 'wf_int',
        #              'dim_func_entities': [],
        #              'pre_ebc_dict': {},
        #              'fem_order': self.FEM_ORDER,
        #              'pre_terms': [pre_term],
        #              'is_exterior': False}

        # # Writing the result pickle file
        # with open(Path(RESULT_DIR / ("translation_rotation_hollow_cylinders_3D_R1/translation_rotation_hollow_cylinders_3D" + "_" + Rx + ".pkl")),
        #           mode="wb") as f:
        #     pickle.dump(args_dict, f)

        # Creation of the two pre_terms
        pre_term_c1 = PreTerm(name='dw_volume_integrate',
                              region_key=('subomega', 300), tag='cst',
                              prefactor=1.0, mat=None,
                              mat_kwargs={'rho': self.rho_1,
                                          'Rc': self.R_Omega})

        pre_term_c2 = PreTerm(name='dw_volume_integrate',
                              region_key=('subomega', 301), tag='cst',
                              prefactor=1.0, mat=None,
                              mat_kwargs={'rho': self.rho_2,
                                          'Rc': self.R_Omega})

        # Creation of the weak form
        args_dict = {'dim': 3,
                      'pre_mesh': str(Path(MESH_DIR / (self.problemName + '_3D_R1.vtk'))),
                      'name': 'wf',
                      'dim_func_entities': [],
                      'pre_ebc_dict': {},
                      'pre_epbc_dict': [],
                      'fem_order': 1,
                      'pre_term': [pre_term_c1, pre_term_c2],
                      'is_exterior': False}

        wf = WeakForm.from_scratch(args_dict=args_dict)
        #wf = WeakForm().from_attr_dict(args_dict)

        args_dict = wf.get_pickable_args_dict()

        with open(Path(RESULT_DIR / (self.problemName + "_3D") / (self.problemName + "_3D.pkl")), mode='wb') as f:
            pickle.dump(args_dict, f)


    def invisible_postprocessing(self, result_pp_R1, result_pp_R2):
        """
        Gets the force vector on the internal test mass without using 3D
        additional mesh files.

        Parameters
        ----------
        result_pp_R1 : RPP
            The first framework's result file.
        result_pp_R2 : RPP
            The second framework's result file.

        Returns
        -------
        Force vector on the internal test mass.

        """

        if self.VERBOSE:
            print(" == INVISIBLE POSTPROCESSING OPERATIONS ==")

        # Using the R2 framework to obtain the vertical gradient
        coors_R2 = result_pp_R2.coors_int
        wf_R2 = result_pp_R2.wf_int
        param = FieldVariable('param', 'parameter', wf_R2.field,
                              primary_var_name=wf_R2.get_unknown_name('cst'))
        param.set_data(coors_R2[:, 0] * result_pp_R2.sol_int)

        expression_IS1_R2 = "ev_grad.{}.subomega300(param)".format(wf_R2.integral.order)
        grad_Phi_IS1_R2 = wf_R2.pb_cst.evaluate(expression_IS1_R2,
                                                var_dict={'param': param})
        F_C1 = grad_Phi_IS1_R2 * self.rho_1 * 2 * np.pi

        print("Force on vertical (z) axis on IS1:", F_C1[1], "N.")



        # # Calculating gradient using the cKD tree method
        # if self.VERBOSE:
        #     print("Potential computed. Calculating gradient...")

        # # Construire un arbre KD
        # tree = cKDTree(coors_3D_R1)

        # # Trouver les voisins les plus proches
        # k = 5  # Réduire k
        # distances, indices = tree.query(coors_3D_R1, k=k+1)

        # # Calculer le gradient en utilisant une pondération par distance
        # gradients = np.zeros_like(coors_3D_R1)

        # for i in range(len(coors_3D_R1)):
        #     neighbor_indices = indices[i, 1:]
        #     neighbor_coords = coors_3D_R1[neighbor_indices]
        #     neighbor_potentials = Phi[neighbor_indices]

        #     centered_coords = neighbor_coords - coors_3D_R1[i]
        #     centered_potentials = neighbor_potentials - Phi[i]

        #     A = centered_coords
        #     b = centered_potentials
        #     gradient, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        #     gradients[i] = gradient

        # for i in range(len(gradients)):
        #     gradients[i] *= dV[i]

        # F_C1 = -np.sum(gradients, axis=0) * self.rho_1


        print(F_C1)
        print("STOP.")


    # Summming the potentials to see if they look as the parallel ones
    def _compare_potential(self, result_pp_R1: RPP, result_pp_R2: RPP):
        """
        This function compares the potential obtained by the sum of R1
        potential and R2 potential with the ones obtained with the
        parallel_hollowCylinders_2D script.

        Parameters
        ----------
        result_pp_R1 : RPP
            Same as above.
        result_pp_R2 : RPP
            Same as above.

        Returns
        -------
        None.

        Note
        ----
        This function is more like a debug function and shuld only be used in
        this class, because it only works when the cylinders are parallel.
        REMEMBER: to work as planned, you must have, in the
        parallel_hollowcylinders_2D's results directory, a consistent
        simulation. The cylinders' dimensions and displacements must be the
        same.

        """

        if self.VERBOSE:
            print("=== COMPARISON WITH PARALLEL CYLINDERS METHOD ===")

        # First part: extraction of the "composed" result
        # Extracting the coordinates of the R2 framework
        coors_R2 = result_pp_R2.coors_int

        # Computing the coordinates coors_R2 but in R1 framework
        coors_R1 = coors_R2 - [self.R_1, self.Z_1]

        # Extracting the potential in R2 coordinates
        sol_int_R2 = result_pp_R2.sol_int

        if self.VERBOSE:
            print("Evaluating the potential and correcting data...")

        # Computing the potential generated by IS1 in the R2 points evaluated in R1 framework
        sol_int_R1 = result_pp_R1.evaluate_at(coors_R1, mode='val', tag='int')

        # If a value is NaN, it means that the point is outside the internal domain
        # Therefore, we must calculate the value in external domain
        for i in range(len(sol_int_R1)):
            if np.isnan(sol_int_R1[i]):
                r = np.sqrt(coors_R1[i, 0]**2 + coors_R1[i, 1]**2)
                r_star = (self.R_Omega**2 / np.abs(r)**2) * r
                cos_theta = coors_R1[i, 0] / r
                sin_theta = coors_R1[i, 1] / r
                kelvin_coors = np.array([[r_star * cos_theta, r_star * sin_theta]])
                sol_int_R1[i] = result_pp_R1.evaluate_at(kelvin_coors, mode='val', tag='ext') # /!\ WORK IN PROGRESS!

        # Computing the sum of the potential
        sol_sum_R2 = sol_int_R2 + sol_int_R1

        # Second part: computation of the result made with parallel_hollowCylinders
        # Note: this value is supposed to be correct
        result_pp_ref = RPP.from_files("../../../parallel_hollowCylinders_2D/data/result/parallel_hollow_cylinders_2D_newton")
        sol_int_ref = result_pp_ref.evaluate_at(coors_R2)
        res = np.abs((sol_sum_R2 - sol_int_ref) / sol_sum_R2)

        if self.VERBOSE:
            print("Plotting the errormap...")

        # Drawing the result
        err_fig, err_ax = plt.subplots(nrows=1, ncols=1)
        im = err_ax.tricontourf(coors_R2[:, 0], coors_R2[:, 1], res, 500)
        err_ax.axis('equal')
        plt.colorbar(im, label='Residual error')

        # Inserting visuals to see the test_masses
        IS1 = patches.Rectangle([self.R_int_1 + self.R_1,
                                 -(self.h_1/2) + self.Z_1],
                                self.R_ext_1-self.R_int_1, self.h_1, color='r',
                                fill=False)
        IS2 = patches.Rectangle([self.R_int_2, -self.h_2/2],
                                self.R_ext_2-self.R_int_2, self.h_2,
                                color='blue', fill=False)
        err_ax.add_patch(IS1)
        err_ax.add_patch(IS2)

        err_ax.set_title("Relative error of the superposition method")
        err_fig.suptitle("Mean relative error:" + str(np.mean(res)))

        # Manually collecting garbage because Python cannot do it himself
        # NOTE: this is important for memory usage
        gc.collect()

        if self.VERBOSE:
            print("Done.")










#%% Testing the class

#@profile

#def test():
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
#Z_1 = 40e-3
R_1 = 0.0

# Second cylinder
R_int_2 = 30.4e-3
R_ext_2 = 34.6975e-3
h_2 = 79.83e-3
rho_2 = 4420
rho_q_2 = 0

# Miscellaneous
FILENAME = 'translation_rotation_hollow_cylinders'
VERBOSE = 1
FEM_ORDER = 2

# Physical parameters
DIM = 2
COORSYS = 'cylindrical'
SOLVER = 'ScipyDirect'

# Mesh size
minSize = 0.0001
maxSize = 0.005
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
mesh_R1_int, mesh_R1_ext, mesh_R2_int, mesh_R2_ext, mesh_3D = FO2PHC.mesh_generation(mesh_3D=False)

print("\n === NEWTONIAN GRAVITY ===")
result_pp_newton = FO2PHC.get_newton_potential(mesh_R1_int, mesh_R1_ext,
                                               mesh_R2_int, mesh_R2_ext)

#FO2PHC.vtk_generation(result_pp_newton[0], result_pp_newton[1])
#FO2PHC.pkl_generation()

# # Creation of a simple vtk file with Meshio
# mesh_3D = meshio.read(Path(MESH_DIR / (FILENAME + "_3D_R1.vtk")))
# coors = mesh_3D.points

# coors_2D = np.column_stack((np.sqrt(coors[:, 0]**2 + coors[:, 1]**2), coors[:, 2]))
# sol_int = result_pp_newton[0].evaluate_at(coors_2D) + result_pp_newton[1].evaluate_at(coors_2D + np.array([R_1, Z_1]))

FO2PHC.invisible_postprocessing(result_pp_newton[0], result_pp_newton[1])

#FO2PHC._compare_potential(result_pp_newton[0], result_pp_newton[1])










#test()