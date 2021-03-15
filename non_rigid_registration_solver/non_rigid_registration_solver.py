"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
import numpy as np
import scipy.sparse as sps
from .base_model import BasicRegistrationSolver
import os
import process_obj_file as p_obj
from scipy.spatial import KDTree
from scipy.sparse.linalg import splu


class RegistrationSolver(BasicRegistrationSolver):

    def __init__(self, cfg):
        super(BasicRegistrationSolver, self).__init__()
        # init the solver
        self.cfg = cfg
        if self.cfg.start_point == 'phase_1':
            self.src_vts, self.src_faces = p_obj.load_obj_file(self.cfg.src_ref_obj)
        elif self.cfg.start_point == 'phase_2':
            self.src_vts, self.src_faces = p_obj.load_obj_file(os.path.join(self.cfg.result_save_dir, self.cfg.ph1_res_name))
        else:
            raise ValueError(f"start point should only be 'phase_1' or 'phase_2 but got: {self.cfg.start_point}'")
        self.dst_vts, self.dst_faces = p_obj.load_obj_file(self.cfg.dst_ref_obj)
        #
        self.src_vts_with_nm, self.src_faces_with_nm = p_obj.add_extra_normal_vertex_for_triangle(self.src_vts, self.src_faces)
        self.dst_vts_with_nm, self.dst_faces_with_nm = p_obj.add_extra_normal_vertex_for_triangle(self.dst_vts, self.dst_faces)
        #
        self.src_mk_vts_idx, self.dst_mk_vts_idx = self.get_mk_vts_indices()
        #
        self.src_mk_slct_mat = self.mk_vts_slct_matrix(self.src_vts_with_nm, mk_vts_idx=self.src_mk_vts_idx)
        self.dst_mk_slct_mat = self.mk_vts_slct_matrix(self.dst_vts_with_nm, mk_vts_idx=self.dst_mk_vts_idx)
        #
        self.src_nbh_map = self.find_neighbors_for_triangle_faces(self.src_faces)
        self.src_nbh_mat = self.build_matrix_neighbors(self.src_nbh_map)
        # build a vertex to face map
        self.src_vts2face_map = self.build_vts2face_map(faces=self.src_faces)
        self.dst_vts2face_map = self.build_vts2face_map(faces=self.dst_faces)
        #
        self.src_a_mat = self.build_matrix_a()
        print("Init non-rigid registration solver successfully!")

    def update_solver(self, new_src_vts: list):
        self.src_vts = new_src_vts
        self.src_vts_with_nm, _ = p_obj.add_extra_normal_vertex_for_triangle(self.src_vts, self.src_faces)
        self.src_a_mat = self.build_matrix_a()
        # reset the solver matrix to None
        self.solver_left_mat = None
        self.solver_right_mat = None
        print("The critical attributes of the solver were updated!")
        return

    def get_mk_vts_indices(self) -> tuple:
        if (not self.cfg.src_marker_txt.endswith('.txt')) or (not self.cfg.dst_marker_txt.endswith('.txt')):
            raise TypeError("The marker vertices info should be loaded from .txt files")
        src_marker_idx = []
        dst_marker_idx = []
        with open(self.cfg.src_marker_txt, 'r') as file:
            lines = file.readlines()
            for line in lines:
                src_marker_idx.append(int(line))
        with open(self.cfg.dst_marker_txt, 'r') as file:
            lines = file.readlines()
            for line in lines:
                dst_marker_idx.append(int(line))
        return src_marker_idx, dst_marker_idx

    def build_phase_1(self):
        """
        Build the Linear equation that need to be optimized: left_mat @ src_vts_with_nm => right_mat
        "@" means the dot product(inner product) of two sparse matrix
        By using LU factorization to solve the linear equation, we could get the optimal deformed "src_vts_with_nm"
        If you can't understand why I construct the linear equation in this way. Try to learn something about:
        Moore-Penrose Pseudo Inverse and Lagrange Multiplier
        """
        s_left, s_right, s_weight = self.build_smooth_optimization_term()
        i_left, i_right, i_weight = self.build_identity_optimization_term()
        cons_left, cons_right, cons_weight = self.build_constraint_optimization_term()
        tmp_left_mat = sps.vstack([s_left, i_left, cons_left], format='csc')
        tmp_right_mat = sps.vstack([s_right, i_right, cons_right], format='csc')
        tmp_weight_mat = sps.block_diag([s_weight, i_weight, cons_weight], format='csc')
        self.solver_left_mat = ((tmp_left_mat.transpose()).dot(tmp_weight_mat)).dot(tmp_left_mat).tocsc()
        self.solver_right_mat = ((tmp_left_mat.transpose()).dot(tmp_weight_mat)).dot(tmp_right_mat).tocsc()
        return

    def build_phase_2(self):
        s_left, s_right, s_weight = self.build_smooth_optimization_term()
        i_left, i_right, i_weight = self.build_identity_optimization_term()
        # closet valid point term should be added in phase 2
        c_left, c_right, c_weight = self.build_closest_valid_point_optimization_term()
        cons_left, cons_right, cons_weight = self.build_constraint_optimization_term()
        tmp_left_mat = sps.vstack([s_left, i_left, c_left, cons_left], format='csc')
        tmp_right_mat = sps.vstack([s_right, i_right, c_right, cons_right], format='csc')
        tmp_weight_mat = sps.block_diag([s_weight, i_weight, c_weight, cons_weight], format='csc')
        self.solver_left_mat = ((tmp_left_mat.transpose()).dot(tmp_weight_mat)).dot(tmp_left_mat).tocsc()
        self.solver_right_mat = ((tmp_left_mat.transpose()).dot(tmp_weight_mat)).dot(tmp_right_mat).tocsc()
        return

    def solve_by_lu_factorization(self):
        inv_left_mat = splu(self.solver_left_mat)
        dense_right_mat = self.solver_right_mat.todense()
        new_src_vts_with_nm = inv_left_mat.solve(dense_right_mat).tolist()
        new_src_vts = new_src_vts_with_nm[:len(self.src_vts)]
        # update the
        self.update_solver(new_src_vts=new_src_vts)
        return

    def non_rigid_registration(self):
        # solve from phase 1 if needed
        if self.cfg.start_point == 'phase_1':
            print("start phase 1 registration!")
            self.build_phase_1()
            self.solve_by_lu_factorization()
            # save phase 1 results
            save_path = os.path.join(self.cfg.result_save_dir, self.cfg.ph1_res_name)
            p_obj.write_obj_file(save_path, vertices=self.src_vts, faces=self.src_faces)
            print("phase 1 registration completed!")
        # solve phase 2
        print("start phase 2 registration!")
        num_ph2_iters = len(self.cfg.wc)
        for idx in range(num_ph2_iters):
            print(f"start phase 2 iteration {idx}")
            self.build_phase_2()
            self.solve_by_lu_factorization()
            save_path = os.path.join(self.cfg.result_save_dir, f"face_phase2_iter{idx}_result.obj")
            p_obj.write_obj_file(save_path, vertices=self.src_vts, faces=self.src_faces)
        # save phase 2 results
        save_path = os.path.join(self.cfg.result_save_dir, self.cfg.ph2_res_name)
        p_obj.write_obj_file(save_path, vertices=self.src_vts, faces=self.src_faces)
        print("phase 2 registration completed!")
        print("Non-rigid registration completed!")
        return

    def build_smooth_optimization_term(self):
        # num of src mesh faces
        num_f = len(self.src_faces)
        # build a [3f, 3f] Identity matrix
        id_mat_3fby3f = sps.identity(n=3 * num_f, format='csc')
        # build the left-side (input) matrix of the linear equation, it's a [n, n] mat
        mat_tmp = (id_mat_3fby3f - self.src_nbh_mat).dot(self.src_a_mat)
        left_mat = (mat_tmp.transpose()).dot(mat_tmp)
        # build the right-side (target) matrix of the linear equation, it's a all-zero matrix with shape [n, 3]
        np_row, np_col, np_val = np.array([]), np.array([]), np.array([])
        right_mat = sps.csc_matrix((np_val, (np_row, np_col)), shape=(len(self.src_vts_with_nm), 3))
        # build the weight matrix
        ws_mat = sps.identity(n=len(self.src_vts_with_nm), format='csc').multiply(2.0 * self.cfg.ws)
        return left_mat, right_mat, ws_mat

    def build_identity_optimization_term(self):
        # build the left-side (input) matrix of the linear equation, it's a [n, n] mat
        left_mat = (self.src_a_mat.transpose()).dot(self.src_a_mat)
        # build the right-side (target) matrix of the linear equation, it's a [n, 3] mat
        np_row, np_col, np_val = np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 1.0, 1.0])
        id_mat_3by3 = sps.csc_matrix((np_val, (np_row, np_col)), shape=(3, 3))
        id_mat_list = []
        for f_idx in range(len(self.src_faces)):
            id_mat_list.append(id_mat_3by3)
        # concatenate num_f [3, 3] matrix vertically together to form a [3f, 3] matrix
        mat_3fby3 = sps.vstack(id_mat_list, format="csc")
        right_mat = (self.src_a_mat.transpose()).dot(mat_3fby3)
        # build the weight matrix
        wi_mat = sps.identity(n=len(self.src_vts_with_nm), format='csc').multiply(2.0 * self.cfg.wi)
        return left_mat, right_mat, wi_mat

    def build_closest_valid_point_optimization_term(self):
        # find closest valid vertices pair first
        vld_clst_vts_pair = self.find_valid_closet_vts()
        # calculate the valid-vertices matrix
        src_vld_mat, dst_vld_mat = self.build_matrix_valid_closet_vts(self.src_vts_with_nm, self.dst_vts_with_nm, vld_clst_vts_pair)
        # build the left-side (input) matrix of the linear equation, it's a [n, n] mat
        left_mat = (src_vld_mat.transpose()).dot(src_vld_mat)
        # build the right-side (target) matrix of the linear equation, it's a [n, 3] mat
        dst_vld_vt_mat = dst_vld_mat.dot(sps.csc_matrix(np.array(self.dst_vts_with_nm)))
        right_mat = (src_vld_mat.transpose()).dot(dst_vld_vt_mat)
        # build the weight matrix
        wc_mat = sps.identity(n=len(self.src_vts_with_nm), format='csc').multiply(2.0 * self.cfg.wc[0])
        self.cfg.wc.pop(0)
        return left_mat, right_mat, wc_mat

    def build_constraint_optimization_term(self):
        # build the left-side (input) matrix of the linear equation, it's a [n+m, n] mat
        left_mat_1 = (self.src_mk_slct_mat.transpose()).dot(self.src_mk_slct_mat)  # [n, n]
        left_mat_2 = self.src_mk_slct_mat  # [m, n]
        left_mat = sps.vstack([left_mat_1, left_mat_2], format='csc')
        # build the right-side (target) matrix of the linear equation, it's a [n+m, 3] mat
        dst_mk_vts_mat = self.dst_mk_slct_mat.dot(sps.csc_matrix(np.array(self.dst_vts_with_nm)))
        right_mat_1 = (self.src_mk_slct_mat.transpose()).dot(dst_mk_vts_mat)  # [n, 3]
        right_mat_2 = dst_mk_vts_mat  # [m, 3]
        right_mat = sps.vstack([right_mat_1, right_mat_2], format='csc')
        # build weight matrix
        cons_mat_1 = sps.identity(n=len(self.src_vts_with_nm), format='csc').multiply(2.0)
        cons_mat_2 = sps.identity(n=len(self.src_mk_vts_idx), format='csc')
        cons_mat = sps.block_diag([cons_mat_1, cons_mat_2], format='csc')
        return left_mat, right_mat, cons_mat

    def build_matrix_a(self) -> sps.csc_matrix:
        """
        Calculate the sparse matrix "A" which described in the equation 9 in the paper
        where A has shape [3f, n]
        mat_k Matrix "K" which is calculated by the function "build_matrix_k"
        mat_tmp Matrix "tmp" which is calculated by the function "build_matrix_tmp"
        """
        mat_k = self.build_matrix_k(self.src_vts_with_nm, self.src_faces_with_nm)
        mat_tmp = self.build_matrix_tmp(self.src_vts_with_nm, self.src_faces_with_nm)
        return mat_k.dot(mat_tmp)

    def find_valid_closet_vts(self):
        vts_kd_tree = KDTree(self.src_vts + self.dst_vts)
        num_orig_n = len(self.src_vts)
        vld_clst_vts_pair = []
        for src_vts_idx, orig_vt in enumerate(self.src_vts):
            query = vts_kd_tree.query(orig_vt, k=self.cfg.num_k, distance_upper_bound=self.cfg.thresh_dist)
            # get the dst-mesh closet valid vertex's index
            dst_vld_vts_list = np.where(np.where(query[1] >= num_orig_n, 1, 0))[0]
            if dst_vld_vts_list.shape[0] == 0:
                # means no valid dst vertices are found, only found vertices of src mesh
                continue
            dst_valid_vts_idx = query[1][dst_vld_vts_list[0]]
            if dst_valid_vts_idx == vts_kd_tree.n:
                # means no valid-vertices is found, see the document of "KDTree.query"
                continue
            # notice the "src_vts_idx" and "dst_valid_vts_idx" should plus 1 to be the actual vertex index in .obj file
            dst_valid_vts_idx = dst_valid_vts_idx - num_orig_n
            src_vt_adj_faces = self.src_vts2face_map[src_vts_idx + 1]
            dst_vt_adj_faces = self.dst_vts2face_map[dst_valid_vts_idx + 1]
            # calculate vertex normal
            src_vt_normal = self.cal_vt_normal_by_adj_faces(self.src_vts_with_nm, self.src_faces_with_nm, src_vt_adj_faces)
            dst_vt_normal = self.cal_vt_normal_by_adj_faces(self.dst_vts_with_nm, self.dst_faces_with_nm, dst_vt_adj_faces)
            dot_product = src_vt_normal @ dst_vt_normal
            if dot_product <= 0.0:
                # means the angle between two normal vector >= 90 degrees
                continue
            vld_clst_vts_pair.append((src_vts_idx, dst_valid_vts_idx))
        return vld_clst_vts_pair

    @staticmethod
    def mk_vts_slct_matrix(vts_with_nm: list, mk_vts_idx: list) -> sps.csc_matrix:
        num_m = len(mk_vts_idx)
        num_n = len(vts_with_nm)
        row = []
        col = []
        value = []
        for row_idx, tmp_mk_vt_idx in enumerate(mk_vts_idx):
            # the marker_vt_idx doesn't need -1
            col_idx = tmp_mk_vt_idx
            row.append(row_idx)
            col.append(col_idx)
            value.append(1.0)
        # build the sparse matrix
        np_row = np.array(row)
        np_col = np.array(col)
        np_val = np.array(value)
        mvs_mat = sps.csc_matrix((np_val, (np_row, np_col)), shape=(num_m, num_n))
        return mvs_mat

    @staticmethod
    def build_matrix_k(vts_with_nm: list, faces_with_nm: list) -> sps.csc_matrix:
        """
        As it stated in paper, we need to build a matrix called "A" to construct the cost function(equation 9 in the paper)
        A = K @ tmp. So the K matrix is a very sparse matrix with shape: (3f, 3f), where f is the number of triangle faces.
        The entries of the K matrix depend on the inverse matrix of the "un-deformed local coordinate system" of each
        triangle face of the target mesh.(equation 4 of the paper)
        Since K is also a sparse matrix with shape: (3f, 3f) which has larger size than tmp matrix.
        To calculate K @ tmp faster, we use scipy.sparse.csc_matrix to be the format of the sparse matrix for both K and tmp
        """
        # obtain the number of triangle faces of mesh
        num_f = len(faces_with_nm)
        # construct "rows, cols, values" array to construct sparse matrix
        row = []
        col = []
        value = []
        for face_idx, face in enumerate(faces_with_nm):
            v0_id, v1_id, v2_id, v3_id = face
            v0, v1, = np.array(vts_with_nm[v0_id - 1], dtype=np.float), np.array(vts_with_nm[v1_id - 1], dtype=np.float)
            v2, v3, = np.array(vts_with_nm[v2_id - 1], dtype=np.float), np.array(vts_with_nm[v3_id - 1], dtype=np.float)
            # build the local coordinate system's axis for each triangle face
            a1, a2, a3 = (v1 - v0).reshape(-1, 1), (v2 - v0).reshape(-1, 1), (v3 - v0).reshape(-1, 1)
            # build the local coordinate system as a square matrix with shape (3, 3)
            coo_sys = np.concatenate([a1, a2, a3], axis=1)
            # calculate the inverse matrix of this 3 by 3 matrix
            coo_sys_inv = np.linalg.inv(coo_sys)
            # transpose the inv
            coo_sys_inv_t = coo_sys_inv.T
            for r in range(3):
                for c in range(3):
                    row.append(face_idx * 3 + r)
                    col.append(face_idx * 3 + c)
                    value.append(coo_sys_inv_t[r][c])
        # build the sparse matrix
        np_row, np_col, np_val = np.array(row), np.array(col), np.array(value)
        k_mat = sps.csc_matrix((np_val, (np_row, np_col)), shape=(3 * num_f, 3 * num_f))
        return k_mat

    @staticmethod
    def build_matrix_tmp(vts_with_nm: list, faces_with_nm: list) -> sps.csc_matrix:
        """
        As it stated in paper, we need to build a matrix called "A" to construct the cost function(equation 9 in the paper)
        A = K @ tmp. So the tmp matrix is a very sparse matrix with shape: (3f, n)
        where f is the number of triangle faces, and n is the number of all vertices(include the normal vertex)
        The entries of tmp matrix only contains (0, 1, -1)
        Since K is also a sparse matrix with shape: (3f, 3f) which has larger size than tmp matrix.
        To calculate K @ tmp faster, we use scipy.sparse.csc_matrix to be the format of the sparse matrix for both K and tmp
        """
        # obtain the number of f and n
        num_n = len(vts_with_nm)
        num_f = len(faces_with_nm)
        # construct "rows, cols, values" array to construct sparse matrix
        row = []
        col = []
        value = []
        row_idx = 0
        for face in faces_with_nm:
            """
            For each face, set 3 rows of the tmp matrix, each row has shape: (1, n)
            """
            v0_id, v1_id, v2_id, v3_id = face
            """
            In each row of those 3 rows, set the value on v0_id to -1.0 and set the value on v1_id(v2, v3) to 1.0
            and the rest are all 0.0. Remember minus 1 for each vertex idx when use
            """
            for vi_id in [v1_id, v2_id, v3_id]:
                # set -1.0 for v0_id
                row.append(row_idx)
                col.append(v0_id - 1)
                value.append(-1.0)
                # set 1.0 for vi_id
                row.append(row_idx)
                col.append(vi_id - 1)
                value.append(1.0)
                # increase the row idx by 1
                row_idx += 1
        # build the sparse matrix
        np_row = np.array(row)
        np_col = np.array(col)
        np_val = np.array(value)
        tmp_mat = sps.csc_matrix((np_val, (np_row, np_col)), shape=(3 * num_f, num_n))
        return tmp_mat

    @staticmethod
    def find_neighbors_for_triangle_faces(faces: list) -> dict:
        """
        build the neighboring list for all triangle faces, notice this step has a O(n) time complexity
        you could accelerate this function by using better algorithm or using multi-thread
        :param faces: all faces of the mesh
        :return: a dict contains{face index: [neighbors faces indices]}
        """
        # build a vertex to face map
        vts2face_map = dict()
        # build a host to neighbor map
        neighbors_map = dict()
        for face_idx, face in enumerate(faces):
            # init an empty list as the value of neighbors_map
            neighbors_map.update({face_idx: []})
            for vt in face:
                if vt not in vts2face_map.keys():
                    vts2face_map.update({vt: [face_idx]})
                else:
                    vts2face_map[vt].append(face_idx)

        for face_idx, face in enumerate(faces):
            for vt in face:
                related_faces = vts2face_map[vt]
                for related_face_idx in related_faces:
                    if related_face_idx not in neighbors_map[face_idx] and related_face_idx != face_idx:
                        neighbors_map[face_idx].append(related_face_idx)
        return neighbors_map

    @staticmethod
    def build_matrix_neighbors(neighbors_map: dict) -> sps.csc_matrix:
        """
        build a matrix called "neighbors" (nbh_mat). By applying nbh_mat @ A @ x, we could obtain the sum of the deformation
        of each triangle face's neighbors. which is described in the equation(11) in the paper.
        nbh_mat has shape [3f, 3f]
        :param neighbors_map: the neighbors map dict of the mesh, which is produced by the function:
        "find_neighbors_for_triangle_faces"
        :return: matrix "nbh"
        """
        # obtain the number of faces
        num_f = len(neighbors_map)
        # construct "rows, cols, values" array to construct sparse matrix
        row = []
        col = []
        value = []
        # start to build the sparse matrix
        for host_face_idx in neighbors_map.keys():
            neighbors_face_list = neighbors_map[host_face_idx]
            num_neighbor_of_host_face = len(neighbors_face_list)
            for neighbors_face_idx in neighbors_face_list:
                # get the idx of the [3,3] matrix
                row_idx_init = host_face_idx * 3
                col_idx_init = neighbors_face_idx * 3
                # set the leading diagonal entries of the [3, 3] matrix to be (1/num_neighbor_of_host_face)
                for i in range(3):
                    row_idx = row_idx_init + i
                    col_idx = col_idx_init + i
                    tmp_value = 1.0 / num_neighbor_of_host_face
                    row.append(row_idx)
                    col.append(col_idx)
                    value.append(tmp_value)
        # build the sparse matrix
        np_row = np.array(row)
        np_col = np.array(col)
        np_val = np.array(value)
        nbh_mat = sps.csc_matrix((np_val, (np_row, np_col)), shape=(num_f * 3, num_f * 3))
        return nbh_mat

    @staticmethod
    def build_vts2face_map(faces: list) -> dict:
        vts2face_map = dict()
        for face_idx, face in enumerate(faces):
            for vt in face:
                if vt not in vts2face_map.keys():
                    vts2face_map.update({vt: [face_idx]})
                else:
                    vts2face_map[vt].append(face_idx)
        return vts2face_map

    @staticmethod
    def cal_vt_normal_by_adj_faces(vts_with_nm: list, faces_with_nm: list, adj_faces_idx: list) -> np.ndarray:
        v3_list = []
        v0_list = []
        for face_idx in adj_faces_idx:
            tmp_v3_idx, tmp_v0_idx = faces_with_nm[face_idx][3] - 1, faces_with_nm[face_idx][0] - 1
            tmp_v3, tmp_v0 = vts_with_nm[tmp_v3_idx], vts_with_nm[tmp_v0_idx]
            v3_list.append(tmp_v3)
            v0_list.append(tmp_v0)
        avg_normal = np.mean(np.array(v3_list) - np.array(v0_list), axis=0)
        return avg_normal

    @staticmethod
    def build_matrix_valid_closet_vts(src_vts_with_nm: list, dst_vts_with_nm: list, vld_clst_vts_pair: list) -> tuple:
        """
        Build a matrix clst_vld_vts_mat. By applying src_clst_vld_vts_mat @ scr_vt, we could obtain the source vertices which
        has the closest valid target vertices, and dst_clst_vld_vts_mat @ dst_vt, we could obtain the target valid vertices.
        src_clst_vld_mat has shape [num_vld_vts, num_src_vts]
        dst_clst_vld_mat has shape [num_vld_vts, num_dst_vts]

        """
        num_src_vts = len(src_vts_with_nm)
        num_dst_vts = len(dst_vts_with_nm)
        num_vld_vts = len(vld_clst_vts_pair)

        src_row, src_col, src_val = [], [], []
        dst_row, dst_col, dst_val = [], [], []

        for row_idx, one_vld_vts_pair in enumerate(vld_clst_vts_pair):
            src_col_idx, dst_col_idx = one_vld_vts_pair

            src_row.append(row_idx)
            src_col.append(src_col_idx)
            src_val.append(1.0)

            dst_row.append(row_idx)
            dst_col.append(dst_col_idx)
            dst_val.append(1.0)
        # build the sparse matrix
        np_src_row, np_src_col, np_src_val = np.array(src_row), np.array(src_col), np.array(src_val)
        np_dst_row, np_dst_col, np_dst_val = np.array(dst_row), np.array(dst_col), np.array(dst_val)
        src_mat = sps.csc_matrix((np_src_val, (np_src_row, np_src_col)), shape=(num_vld_vts, num_src_vts))
        dst_mat = sps.csc_matrix((np_dst_val, (np_dst_row, np_dst_col)), shape=(num_vld_vts, num_dst_vts))
        return src_mat, dst_mat
