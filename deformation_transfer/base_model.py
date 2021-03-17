"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from abc import ABC, abstractmethod
import scipy.sparse as sps
import argparse


class BasicDeformationTransferSolver(ABC):

    def __init__(self):
        """
        Init the solver, set needed attributes to None at first
        """
        '''
        Attributes that doesn't need to be updated 
        '''
        self.cfg: argparse.Namespace = None
        # reference vertices of source mesh
        self.src_ref_vts: list = None
        # reference vertices of target mesh
        self.dst_ref_vts: list = None

        # faces of two mesh
        self.src_faces: list = None
        self.dst_faces: list = None
        # vertices and triangle faces after adding normal vector vertex
        self.src_ref_vts_with_nm: list = None
        self.src_faces_with_nm: list = None
        #
        self.crspd_src_faces_with_nm: list = None
        self.crspd_dst_faces: list = None
        self.n_crspd_dst_faces_with_nm: list = None
        #
        self.mat_a_crspd: sps.csc_matrix = None  # only correspond with reference target mesh, won't be changed easily
        self.mat_a_not_crspd: sps.csc_matrix = None  # won't be changed easily as well
        self.mat_c_not_crspd: sps.csc_matrix = None
        self.solver_left_mat: sps.csc_matrix = None  # only correspond with matrix A, won't be changed easily

        '''
        Attributes that needed to be updated frequently
        should be updated for every frame if process mesh-sequence deformation transfer
        '''
        # deformed vertices of source mesh
        self.src_def_vts: list = None  #
        self.src_def_vts_with_nm: list = None
        self.solver_right_mat: sps.csc_matrix = None  # correspond with the deformation of source mesh

    @abstractmethod
    def build_problem(self, src_def_vts: list):
        """
        Build the linear-optimization problem, since in the paper, the deformation transfer is equivalent to a
        linear-optimization problem.
        """
        pass

    @abstractmethod
    def solve_problem(self) -> list:
        """
        Solve the linear-optimization problem with LU factorization
        """
        pass
