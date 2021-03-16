"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
import os
from correspondence_finder import CorrespondenceFinder
import json
from config import get_correspondence_finder_args

"""
This script find the correspondence triangle faces of two meshes with different topology.
The correspondence a many-to-many mapping.
The indices of the correspondent faces will be saved into a json file
"""

if __name__ == "__main__":
    cfg = get_correspondence_finder_args()
    finder = CorrespondenceFinder(cfg)
    src_crspd_faces_indices, dst_crspd_faces_indices = finder.find_correspondence()
    finder.check_crspd_faces_distance(src_crspd_faces_indices, dst_crspd_faces_indices)
    data_dict = {'src': src_crspd_faces_indices, 'dst': dst_crspd_faces_indices}
    # save the results into json file
    save_path = os.path.join(cfg.save_dir, cfg.save_name)
    with open(save_path, 'w') as f:
        json.dump(data_dict, f)
    print(f'Successfully save correspondence into: {save_path}')
