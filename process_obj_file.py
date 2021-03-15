"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""

import numpy as np
import copy

def load_obj_file(path: str):
    """
    Load obj file
    load the .obj format mesh file with square or triangle faces
    return the vertices list and faces list
    """
    if path.endswith('.obj'):
        file = open(path, 'r')
        lines = file.readlines()
        vertices = []
        faces = []
        for line in lines:
            if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'):
                line_split = line.split(" ")
                # ver = line_split[1:4]
                ver = [each for each in line_split[1:] if each != '']
                ver = [float(v) for v in ver]
                vertices.append(ver)
            else:
                if line.startswith('f'):
                    line_split = line.split(" ")
                    if '/' in line:
                        tmp_faces = line_split[1:]
                        f = []
                        for tmp_face in tmp_faces:
                            f.append(int(tmp_face.split('/')[0]))
                        faces.append(f)
                    else:
                        face = line_split[1:]
                        face = [int(fa) for fa in face]
                        faces.append(face)
        return vertices, faces
    else:
        raise ValueError('Wrong file format，not a correct .obj mesh file!')


def write_obj_file(file_name_path: str, vertices: list, faces: list):
    """
    write the obj file to the specific path
    file_name_path: path to write the obj file
    vertices: list
    faces:  list
    """
    with open(file_name_path, 'w') as f:
        for v in vertices:
            # print(v)
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            if len(face) == 4:
                f.write("f {} {} {} {}\n".format(face[0], face[1], face[2], face[3]))
            if len(face) == 3:
                f.write("f {} {} {}\n".format(face[0], face[1], face[2]))
    print(f"successfully write the obj file at: {file_name_path}")
    return


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
    # # deep copy the input to prevent the original list from being changed
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
