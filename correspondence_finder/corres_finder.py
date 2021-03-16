"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from scipy.spatial import KDTree
import process_obj_file as p_obj
import copy
import numpy as np


class CorrespondenceFinder(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.src_reg_vts, self.src_reg_faces = p_obj.load_obj_file(cfg.src_reg_obj)
        self.dst_ref_vts, self.dst_ref_faces = p_obj.load_obj_file(cfg.dst_ref_obj)

        self.src_reg_vts_with_nm, self.src_reg_faces_with_nm = \
            p_obj.add_extra_normal_vertex_for_triangle(self.src_reg_vts, self.src_reg_faces)
        self.dst_ref_vts_with_nm, self.dst_ref_faces_with_nm = \
            p_obj.add_extra_normal_vertex_for_triangle(self.dst_ref_vts, self.dst_ref_faces)

        self.src_centroid_list = self.cal_triangle_face_centroid(self.src_reg_vts, self.src_reg_faces)
        self.dst_centroid_list = self.cal_triangle_face_centroid(self.dst_ref_vts, self.dst_ref_faces)

        self.src_kd_tree = KDTree(self.src_centroid_list)
        self.dst_kd_tree = KDTree(self.dst_centroid_list)
        print("Successfully init the correspondence finder!")

    def find_correspondence(self) -> tuple:
        # find correspondence
        final_crspd_src_faces_idx = []
        final_crspd_dst_faces_idx = []
        # config for kd_tree query
        num_k = self.cfg.num_k
        thresh_dist = self.cfg.thresh_dist
        # find dst's nearest pairing triangle face for each src triangle
        for src_face_idx, src_centroid in enumerate(self.src_centroid_list):
            query = self.dst_kd_tree.query(np.array(src_centroid), k=num_k, distance_upper_bound=thresh_dist)
            nearest_dst_faces_list = query[1].tolist()
            # means no valid dst centroid of triangle faces are found
            if len(nearest_dst_faces_list) == 0: continue
            tmp_dst_face_list = []
            for dst_valid_face_idx in nearest_dst_faces_list:
                if dst_valid_face_idx == self.dst_kd_tree.n: continue
                # compare the normal vector
                angle_larger_than_90 = self.compare_normal_vector(src_face_idx, dst_valid_face_idx)
                # means the angle between two normal vector >= 90 degrees
                if angle_larger_than_90: continue
                tmp_dst_face_list.append(dst_valid_face_idx)
            # add closet corresponded faces into result lists
            if len(tmp_dst_face_list) != 0:
                final_crspd_src_faces_idx.append(src_face_idx)
                final_crspd_dst_faces_idx.append(tmp_dst_face_list[0])

        # find src's nearest pairing triangle face for each dst triangle
        for dst_face_idx, dst_centroid in enumerate(self.dst_centroid_list):
            query = self.src_kd_tree.query(np.array(dst_centroid), k=num_k, distance_upper_bound=thresh_dist)
            nearest_src_faces_list = query[1].tolist()
            # means no valid dst centroid of triangle faces are found
            if len(nearest_src_faces_list) == 0: continue
            tmp_src_face_list = []
            for src_valid_face_idx in nearest_src_faces_list:
                if src_valid_face_idx == self.src_kd_tree.n: continue
                # compare the normal vector
                angle_larger_than_90 = self.compare_normal_vector(src_valid_face_idx, dst_face_idx)
                if angle_larger_than_90: continue
                tmp_src_face_list.append(src_valid_face_idx)
            if len(tmp_src_face_list) != 0:
                final_crspd_src_faces_idx.append(tmp_src_face_list[0])
                final_crspd_dst_faces_idx.append(dst_face_idx)

        return final_crspd_src_faces_idx, final_crspd_dst_faces_idx

    def check_crspd_faces_distance(self, crspd_src_f_indices: list, crspd_dst_f_indices: list):
        for src_f_idx, dst_f_idx in zip(crspd_src_f_indices, crspd_dst_f_indices):
            src_centroid = self.src_centroid_list[src_f_idx]
            dst_centroid = self.dst_centroid_list[dst_f_idx]
            dist = np.linalg.norm(np.array(src_centroid) - np.array(dst_centroid))
            if dist > self.cfg.thresh_dist:
                print(src_f_idx, dst_f_idx)
                raise ValueError("Wrong triangle pair!")
        print("All triangle pairs are correct!")
        return

    @staticmethod
    def cal_triangle_face_centroid(vts: list, faces: list):
        centroid_list = []
        for face in faces:
            v0_idx, v1_idx, v2_idx = face[0] - 1, face[1] - 1, face[2] - 1
            v0, v1, v2 = vts[v0_idx], vts[v1_idx], vts[v2_idx]
            np_triangle_vts = np.array([v0, v1, v2])
            centroid = np.mean(np_triangle_vts, axis=0).tolist()
            centroid_list.append(centroid)
        return centroid_list

    def compare_normal_vector(self, crspd_src_f_idx: int, crspd_dst_f_idx: int) -> bool:
        crspd_src_face_with_nm = self.src_reg_faces_with_nm[crspd_src_f_idx]
        crspd_dst_face_with_nm = self.dst_ref_faces_with_nm[crspd_dst_f_idx]

        src_v0_idx, src_v3_idx = crspd_src_face_with_nm[0] - 1, crspd_src_face_with_nm[3] - 1
        dst_v0_idx, dst_v3_idx = crspd_dst_face_with_nm[0] - 1, crspd_dst_face_with_nm[3] - 1
        # obtain normal vector of triangle face
        src_normal = np.array(self.src_reg_vts_with_nm[src_v3_idx]) - np.array(self.src_reg_vts_with_nm[src_v0_idx])
        dst_normal = np.array(self.dst_ref_vts_with_nm[dst_v3_idx]) - np.array(self.dst_ref_vts_with_nm[dst_v0_idx])
        dot_product = src_normal @ dst_normal
        larger_than_90 = True if (dot_product <= 0) else False
        return larger_than_90

