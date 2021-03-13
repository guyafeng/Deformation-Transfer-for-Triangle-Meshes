"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""


def load_obj_file(path):
    """Load obj file
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
        print('Wrong file format，not a correct .obj mesh file!')
        return None, None


def write_obj_file(file_name_path: str, vertices: list, faces: list):
    """write the obj file to the specific path
       file_name_path: path to write the obj file
       vertices: list
       faces: 面 list
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
