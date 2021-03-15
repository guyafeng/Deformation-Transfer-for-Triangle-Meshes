"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from abc import ABC, abstractmethod
import scipy.sparse as sps
import argparse


class BasicRegistrationSolver(ABC):

    def __init__(self):
        self.cfg: argparse.Namespace = None
        # original vertices and faces loaded from .obj files
        self.src_vts: list = None  # should be updated after every-time optimization
        self.dst_vts: list = None
        self.src_faces: list = None
        self.dst_faces: list = None
        #
        self.src_vts_with_nm: list = None  # should be updated after every-time optimization
        self.dst_vts_with_nm: list = None
        self.src_faces_with_nm: list = None
        self.dst_faces_with_nm: list = None
        # marker vertices list
        self.src_mk_vts_idx: list = None
        self.dst_mk_vts_idx: list = None
        # marker vertices selection matrix
        self.src_mk_slct_mat: sps.csc_matrix = None
        self.dst_mk_slct_mat: sps.csc_matrix = None
        # neighbors map and neighboring deformation summing matrix
        self.src_nbh_map: dict = None
        self.src_nbh_mat: sps.csc_matrix = None
        # vertices-face map indicating which faces consists of a specified vertex
        self.src_vts2face_map: dict = None
        self.dst_vts2face_map: dict = None
        # the A matrix is used to calculate the deformation gradients of a mesh as it stated in the paper
        self.src_a_mat: sps.csc_matrix = None  # should be updated after every-time optimization
        # the final solving matrix of the linear equation
        self.solver_left_mat: sps.csc_matrix = None
        self.solver_right_mat: sps.csc_matrix = None

    @abstractmethod
    def update_solver(self, new_src_vts: list):
        """
        update some necessary attributes after every-time optimization
        """
        pass

    @abstractmethod
    def build_phase_1(self):
        """
        build the input, target matrix for LU factorization of phase 1.
        """
        pass

    @abstractmethod
    def build_phase_2(self):
        """
        build the input, target matrix for LU factorization of phase 2.
        """
        pass

    @abstractmethod
    def solve_by_lu_factorization(self):
        """
        solve the non-rigid registration of phase-1 by LU factorization
        """
        pass

    @abstractmethod
    def non_rigid_registration(self):
        """
        the complete pipeline of the non-rigid registration solver
        """
        pass



