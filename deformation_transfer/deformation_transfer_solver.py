"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from .base_model import BasicDeformationTransferSolver
import process_obj_file as p_obj
import json
import scipy.sparse as sps
import numpy as np
import scipy
from scipy.sparse.linalg import splu


class DeformationTransferSolver(BasicDeformationTransferSolver):

    def __init__(self, cfg):
        super(DeformationTransferSolver, self).__init__()
        self.cfg = cfg
        #
        self.src_ref_vts, self.src_faces = p_obj.load_obj_file(cfg.src_ref_obj)
        self.dst_ref_vts, self.dst_faces = p_obj.load_obj_file(cfg.dst_ref_obj)
        #
        self.src_ref_vts_with_nm, self.src_faces_with_nm = \
            p_obj.add_extra_normal_vertex_for_triangle(self.src_ref_vts, self.src_faces)
        #
        if not cfg.same_topology:
            with open(cfg.crspd_file, 'r') as f:
                crspd_data = json.load(f)
            # get not corresponded faces indices for target mesh
            n_crspd_dst_faces_indices = list(set(range(len(self.dst_faces))) - set(crspd_data['dst']))
            # load correspond and not-correspond faces
            self.crspd_src_faces_with_nm = [self.src_faces_with_nm[idx] for idx in crspd_data['src']]
            # don't need to care the not-crspd faces of source mesh
            self.crspd_dst_faces = [self.dst_faces[idx] for idx in crspd_data['dst']]
            self.n_crspd_dst_faces = [self.dst_faces[idx] for idx in n_crspd_dst_faces_indices]
            # construct matrices that won't be updated frequently
            self.mat_a_crspd = self.construct_matrix_a(mode='crspd')
            self.mat_a_not_crspd = self.construct_matrix_a(mode='n_crspd')
            self.mat_c_not_crspd = self.construct_matrix_c(mode='n_crspd')
            tmp_left_mat = sps.vstack([self.mat_a_crspd, self.mat_a_not_crspd], format='csc')
            self.solver_left_mat = (tmp_left_mat.transpose()).dot(tmp_left_mat).tocsc()
        else:
            # if source and target mesh has same topology, things are much simpler
            self.crspd_src_faces_with_nm = self.src_faces_with_nm
            self.crspd_dst_faces = self.dst_faces
            self.mat_a_crspd = self.construct_matrix_a(mode='crspd')
            self.solver_left_mat = (self.mat_a_crspd.transpose()).dot(self.mat_a_crspd).tocsc()
        self.inv_left_mat = splu(self.solver_left_mat)
        print("Init deformation transfer solver successfully!")

    def build_problem(self, src_def_vts: list) -> sps.csc_matrix:
        self.src_def_vts = src_def_vts
        self.src_def_vts_with_nm, _ = \
            p_obj.add_extra_normal_vertex_for_triangle(self.src_def_vts, self.src_faces)
        mat_c_crspd = self.construct_matrix_c(mode='crspd')
        if not self.cfg.same_topology:
            tmp_right_mat = sps.vstack([mat_c_crspd, self.mat_c_not_crspd], format='csc')
            tmp_left_mat = sps.vstack([self.mat_a_crspd, self.mat_a_not_crspd], format='csc')
            solver_right_mat = (tmp_left_mat.transpose()).dot(tmp_right_mat)
        else:
            solver_right_mat = (self.mat_a_crspd.transpose()).dot(mat_c_crspd)
        return solver_right_mat

    def solve_problem(self, solver_right_mat: sps.csc_matrix) -> list:
        dense_right_mat = solver_right_mat.todense()
        dst_def_vts = self.inv_left_mat.solve(dense_right_mat).tolist()
        return dst_def_vts

    def construct_matrix_a(self, mode: str='crspd') -> sps.csc_matrix:
        """
        Construct the matrix A which described in paper, by applying mat_a @ vts, we could obtain the deformation
        gradients of the input triangle faces. Here we follow the methodology that described in the sumner's thesis:
        "Mesh Modification Using Deformation Gradients" Chapter 3.
        :return: a sparse matrix with shape [3f, n]
        """
        vts = self.dst_ref_vts
        if mode == 'crspd':
            faces = self.crspd_dst_faces
        elif mode == 'n_crspd':
            faces = self.n_crspd_dst_faces
        else:
            raise ValueError("Wrong mode for construct matrix A")
        num_n = len(vts)
        num_f = len(faces)
        row, col, val = [], [], []
        for face_idx, face in enumerate(faces):
            v0_id, v1_id, v2_id = face
            # QR factorization, which referred to the equation (3.11) of Sumner's thesis in page:55
            r_inv_q_trans, _ = self.qr_factorization_one_face(ref_vts=vts, one_face=face)  # r_inv_q_trans: [2, 3]
            # the code below referred to page:65 of Sumner's thesis "Mesh Modification Using Deformation Gradients"
            elem_a, elem_b, elem_c = r_inv_q_trans[0][0], r_inv_q_trans[0][1], r_inv_q_trans[0][2]
            elem_d, elem_e, elem_f = r_inv_q_trans[1][0], r_inv_q_trans[1][1], r_inv_q_trans[1][2]
            # build a [3,3] matrix
            local_mat_elem = [[elem_a, elem_d, -1.0 * (elem_a + elem_d)],
                              [elem_b, elem_e, -1.0 * (elem_b + elem_e)],
                              [elem_c, elem_f, -1.0 * (elem_c + elem_f)]]
            # insert the [3, 3] matrix into a huge sparse matrix
            for local_r_idx, row_idx in enumerate([face_idx * 3, face_idx * 3 + 1, face_idx * 3 + 2]):
                for local_c_idx, col_idx in enumerate([v1_id - 1, v2_id - 1, v0_id - 1]):
                    row.append(row_idx)
                    col.append(col_idx)
                    val.append(local_mat_elem[local_r_idx][local_c_idx])
        # build the final sparse matrix
        np_row, np_col, np_val = np.array(row), np.array(col), np.array(val)
        sps_mat_a = sps.csc_matrix((np_val, (np_row, np_col)), shape=(3 * num_f, num_n))
        return sps_mat_a

    def construct_matrix_c(self, mode: str = 'crspd') -> sps.csc_matrix:
        """
        construct the deformation gradient of the target mesh. it's a matrix with shape [3f, 3]
        Here we follow the methodology that described in the sumner's thesis:
        "Mesh Modification Using Deformation Gradients", Chapter 3. Mainly the equation (3.15), (3.16) in page 62.
        :return: a sparse matrix with shape [3f, 3]
        """
        transpose_affine_list = []
        if mode == "crspd":
            for tmp_src_face_with_nm, tmp_dst_face in zip(self.crspd_src_faces_with_nm, self.crspd_dst_faces):
                # calculate the local affine transform (i.e. deformation gradient) of the source triangle
                src_local_affine = self.cal_deformation_gradient(self.src_ref_vts_with_nm, self.src_def_vts_with_nm,
                                                                 tmp_src_face_with_nm)
                # build dst portion
                r_inv_q_trans, dst_w = self.qr_factorization_one_face(self.dst_ref_vts, tmp_dst_face)  # [2, 3]
                final_local_affine = src_local_affine @ (dst_w @ r_inv_q_trans)
                final_local_affine_transpose = final_local_affine.T
                transpose_affine_list.append(final_local_affine_transpose)
        elif mode == 'n_crspd':
            for tmp_dst_face in self.n_crspd_dst_faces:
                # for not-corresponded faces, the target deformation gradient is an identity matrix
                src_local_affine = np.identity(n=3)
                # build dst portion
                r_inv_q_trans, dst_w = self.qr_factorization_one_face(self.dst_ref_vts, tmp_dst_face)  # [2, 3]
                final_local_affine = src_local_affine @ (dst_w @ r_inv_q_trans)
                final_local_affine_transpose = final_local_affine.T
                transpose_affine_list.append(final_local_affine_transpose)
        else:
            raise ValueError("Wrong mode for construct matrix C.")

        mat_c = np.concatenate(transpose_affine_list, axis=0)
        sps_mat_c = sps.csc_matrix(mat_c)
        return sps_mat_c

    @staticmethod
    def qr_factorization_one_face(ref_vts: list, one_face: list) -> tuple:
        """
        Calculate the R.inv @ Q.T, which referred to the equation (3.11) of Sumner's thesis in page:55
        :return two matrix with shape [2, 3] and [3, 2]
        """
        v0_id, v1_id, v2_id = one_face
        v0 = np.array(ref_vts[v0_id - 1], dtype=np.float)
        v1 = np.array(ref_vts[v1_id - 1], dtype=np.float)
        v2 = np.array(ref_vts[v2_id - 1], dtype=np.float)
        # get the 2 edge of one triangle face
        a1 = (v1 - v0).reshape(-1, 1)
        a2 = (v2 - v0).reshape(-1, 1)
        # form matrix Wj in page:55 of thesis
        w = np.concatenate([a1, a2], axis=1)  # [3, 2]
        q, r = scipy.linalg.qr(w)  # q[3, 3], r[3, 2]
        r_inv_q_trans = np.linalg.inv(r[0:2, 0:2]) @ (q[0:3, 0:2].transpose())  # [2, 3]
        return r_inv_q_trans, w

    @staticmethod
    def cal_deformation_gradient(src_ref_vts_with_nm: list, src_def_vts_with_nm: list, one_face_with_nm: list)->np.ndarray:
        """
        Calculate the deformation gradient of a triangle face based on the methodology described in Sumner's paper:
        "Deformation Transfer for Triangle Meshes". Mainly referred to the equation (4) in Chapter 3.
        :return:
        """
        v0_id, v1_id, v2_id, v3_id = one_face_with_nm
        # get the vertices of the corresponded reference triangle
        src_v0_ref = np.array(src_ref_vts_with_nm[v0_id - 1], dtype=np.float)
        src_v1_ref = np.array(src_ref_vts_with_nm[v1_id - 1], dtype=np.float)
        src_v2_ref = np.array(src_ref_vts_with_nm[v2_id - 1], dtype=np.float)
        src_v3_ref = np.array(src_ref_vts_with_nm[v3_id - 1], dtype=np.float)
        # get the vertices of the corresponded deformed triangle
        src_v0_def = np.array(src_def_vts_with_nm[v0_id - 1], dtype=np.float)
        src_v1_def = np.array(src_def_vts_with_nm[v1_id - 1], dtype=np.float)
        src_v2_def = np.array(src_def_vts_with_nm[v2_id - 1], dtype=np.float)
        src_v3_def = np.array(src_def_vts_with_nm[v3_id - 1], dtype=np.float)
        # build the local coordinate system's axis for reference triangle face
        src_a1_ref = (src_v1_ref - src_v0_ref).reshape(-1, 1)
        src_a2_ref = (src_v2_ref - src_v0_ref).reshape(-1, 1)
        src_a3_ref = (src_v3_ref - src_v0_ref).reshape(-1, 1)
        # build the local coordinate system's axis for deformed triangle face
        src_a1_def = (src_v1_def - src_v0_def).reshape(-1, 1)
        src_a2_def = (src_v2_def - src_v0_def).reshape(-1, 1)
        src_a3_def = (src_v3_def - src_v0_def).reshape(-1, 1)
        # build the local coordinate system as a square matrix with shape (3, 3)
        coo_sys_ref = np.concatenate([src_a1_ref, src_a2_ref, src_a3_ref], axis=1)
        coo_sys_def = np.concatenate([src_a1_def, src_a2_def, src_a3_def], axis=1)
        # calculate the local affine transform (i.e. deformation gradient) of the triangle face
        deform_grad = coo_sys_def @ np.linalg.inv(coo_sys_ref)
        return deform_grad
