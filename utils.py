"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
import scipy
import scipy.sparse as sps
import numpy as np
import copy


def add_extra_normal_vertex_for_triangle(vertices: list, faces: list):
    """
    add the fourth vertex for each triangle face, we call the fourth vertex as the "normal-vertex"
    we need to use the four vertices of one triangle face to calculate the deformation gradient of target mesh.
    Here we follow the methodology that described in the sumner's thesis:
    "Mesh Modification Using Deformation Gradients", Chapter 3.
    :param vertices:mesh模型的顶点列表
    :param faces:mesh模型的三角面
    :return:v_new, f_new 新的顶点和三角面列表
    """
    # 深拷贝创建新的对象，防止对原有变量引用做修改
    v_new = copy.deepcopy(vertices)
    f_new = copy.deepcopy(faces)
    # 注意face中的vertex序号是1开始的
    for face in f_new:
        v0_id, v1_id, v2_id = face
        v0, v1, v2 = np.array(vertices[v0_id - 1], dtype=np.float), np.array(vertices[v1_id - 1], dtype=np.float), np.array(vertices[v2_id - 1], dtype=np.float)
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2) / np.linalg.norm(np.cross(e1, e2))
        vn = v0 + normal
        vn_list_form = vn.tolist()
        # append the new normal vertex directly
        v_new.append(vn_list_form)
        # 获取新插入顶点的序号，不用减去1
        vn_idx = len(v_new)
        # 将新序号，插入f_new的对应的face中
        face.append(vn_idx)
    return v_new, f_new


def construct_mat_a(vts: list, faces: list) -> sps.csc_matrix:
    """
    construct the matrix A of the paper, by applying mat_a @ vts, we could obtain the deformation gradients of the
    mesh. Here we follow the methodology that described in the sumner's thesis:
    "Mesh Modification Using Deformation Gradients" Chapter 3.
    :param vts: vertices of mesh
    :param faces: triangle faces of mesh
    :return: a sparse matrix with shape [3f, n]
    """
    num_n = len(vts)
    num_f = len(faces)
    row = []
    col = []
    val = []
    for face_idx, face in enumerate(faces):
        v0_id, v1_id, v2_id = face
        v0 = np.array(vts[v0_id - 1], dtype=np.float)
        v1 = np.array(vts[v1_id - 1], dtype=np.float)
        v2 = np.array(vts[v2_id - 1], dtype=np.float)

        a1 = (v1 - v0).reshape(-1, 1)
        a2 = (v2 - v0).reshape(-1, 1)

        w = np.concatenate([a1, a2], axis=1)  # [3, 2]
        q, r = scipy.linalg.qr(w)  # q[3, 3], r[3, 2]
        r_inv_q_trans = np.linalg.inv(r[0:2, 0:2]) @ (q[0:3, 0:2].transpose())  # [2, 3]

        elem_a, elem_b, elem_c = r_inv_q_trans[0][0], r_inv_q_trans[0][1], r_inv_q_trans[0][2]
        elem_d, elem_e, elem_f = r_inv_q_trans[1][0], r_inv_q_trans[1][1], r_inv_q_trans[1][2]

        local_mat_elem = [[elem_a, elem_d, -1.0 * (elem_a + elem_d)],
                          [elem_b, elem_e, -1.0 * (elem_b + elem_e)],
                          [elem_c, elem_f, -1.0 * (elem_c + elem_f)]]

        for local_r_idx, row_idx in enumerate([face_idx * 3, face_idx * 3 + 1, face_idx * 3 + 2]):
            for local_c_idx, col_idx in enumerate([v1_id - 1, v2_id - 1, v0_id - 1]):
                row.append(row_idx)
                col.append(col_idx)
                val.append(local_mat_elem[local_r_idx][local_c_idx])

    np_row = np.array(row)
    np_col = np.array(col)
    np_val = np.array(val)

    a_mat = sps.csc_matrix((np_val, (np_row, np_col)), shape=(3 * num_f, num_n))
    return a_mat


def construct_mat_c(src_ref_vts_with_nm: list, src_def_vts_with_nm: list,
                    dst_ref_vts: list,
                    src_faces_with_nm: list,
                    dst_faces: list) -> sps.csc_matrix:
    """
    construct the deformation gradient of the target mesh. it's a matrix with shape [3f, 3]
    Here we follow the methodology that described in the sumner's thesis:
    "Mesh Modification Using Deformation Gradients", Chapter 3.
    :param src_ref_vts_with_nm: source reference mesh vertices, include the normal vertex
    :param src_def_vts_with_nm: source deformed mesh vertices, include the normal vertex
    :param src_faces_with_nm: source mesh faces, include the normal vertex
    :param dst_ref_vts: target reference mesh vertices, no normal vertex
    :param dst_faces: target mesh faces, no normal vertex
    :return:
    """
    transpose_affine_list = []
    for face_idx, face_tuple in enumerate(zip(src_faces_with_nm, dst_faces)):
        tmp_src_face, tmp_dst_face = face_tuple[0], face_tuple[1]
        src_v0_id, src_v1_id, src_v2_id, src_v3_id = tmp_src_face
        dst_v0_id, dst_v1_id, dst_v2_id = tmp_dst_face
        # get the vertices of the reference triangle
        src_v0_ref = np.array(src_ref_vts_with_nm[src_v0_id - 1], dtype=np.float)
        src_v1_ref = np.array(src_ref_vts_with_nm[src_v1_id - 1], dtype=np.float)
        src_v2_ref = np.array(src_ref_vts_with_nm[src_v2_id - 1], dtype=np.float)
        src_v3_ref = np.array(src_ref_vts_with_nm[src_v3_id - 1], dtype=np.float)
        # get the vertices of the deformed triangle
        src_v0_def = np.array(src_def_vts_with_nm[src_v0_id - 1], dtype=np.float)
        src_v1_def = np.array(src_def_vts_with_nm[src_v1_id - 1], dtype=np.float)
        src_v2_def = np.array(src_def_vts_with_nm[src_v2_id - 1], dtype=np.float)
        src_v3_def = np.array(src_def_vts_with_nm[src_v3_id - 1], dtype=np.float)
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
        # calculate the local affine transform of the source triangle
        src_local_affine = coo_sys_def @ np.linalg.inv(coo_sys_ref)

        # build dst portion
        dst_v0_ref = np.array(dst_ref_vts[dst_v0_id - 1], dtype=np.float)
        dst_v1_ref = np.array(dst_ref_vts[dst_v1_id - 1], dtype=np.float)
        dst_v2_ref = np.array(dst_ref_vts[dst_v2_id - 1], dtype=np.float)

        dst_a1_ref = (dst_v1_ref - dst_v0_ref).reshape(-1, 1)
        dst_a2_ref = (dst_v2_ref - dst_v0_ref).reshape(-1, 1)

        dst_w = np.concatenate([dst_a1_ref, dst_a2_ref], axis=1)  # [3, 2]
        q, r = scipy.linalg.qr(dst_w)  # q[3, 3], r[3, 2]

        r_inv_q_trans = np.linalg.inv(r[0:2, 0:2]) @ (q[0:3, 0:2].transpose())  # [2, 3]
        final_local_affine = src_local_affine @ (dst_w @ r_inv_q_trans)
        final_local_affine_transpose = final_local_affine.T
        transpose_affine_list.append(final_local_affine_transpose)

    mat_c = np.concatenate(transpose_affine_list, axis=0)
    sps_mat_c = sps.csc_matrix(mat_c)
    return sps_mat_c