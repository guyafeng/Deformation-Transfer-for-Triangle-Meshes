"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from abc import ABC, abstractmethod
import scipy.sparse as sps
from scipy.spatial import KDTree


class BasicCorresSolver(ABC):

    def __init__(self, cfg):
        self.cfg = cfg

        self.src_vts: list = None
        self.dst_vts: list = None
        self.src_faces: list = None
        self.dst_faces: list = None

        self.src_vts_with_nm: list = None
        self.dst_vts_with_nm: list = None
        self.src_faces_with_nm: list = None
        self.dst_faces_with_nm: list = None

        self.src_mk_vts_idx: list = None
        self.dst_mk_vts_idx: list = None

        self.src_mk_slct_mat: sps.csc_matrix = None
        self.dst_mk_slct_mat: sps.csc_matrix = None
        self.src_mk_vts: sps.csc_matrix = None
        self.dst_mk_vts: sps.csc_matrix = None

        self.vts_kd_tree: KDTree = None
        self.fs_kd_tree: KDTree = None

    @abstractmethod
    def init_solver(self):
        """
        init the correspondence solver
        """
        pass

    @abstractmethod
    def load_mk_vts_from_txt(self):
        """
        load marker vertices from .txt file
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
    def solve_phase_1(self):
        """
        solve the non-rigid registration of phase-1 by LU factorization
        """
        pass

    @abstractmethod
    def solve_phase_2(self):
        """
        solve the non-rigid registration of phase-2 by LU factorization
        """
        pass

    @abstractmethod
    def find_correspondence(self):
        """
        find the correspondence faces of two meshes with different topology
        """
        pass

    @abstractmethod
    def mk_vts_slct_matrix(self):
        """
        build a sparse matrix mk_vts_slct_mat. By applying mk_vts_slct_mat @ vts. We can "select" the
        marker vertices from the whole vertices matrix.
        """
        pass