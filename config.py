"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
import argparse

'''
Config parser of non-rigid registration solver
'''


def get_registration_solver_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_point', type=str, default='phase_2')
    parser.add_argument('--src_ref_obj', type=str, default='./data/face-poses/face-reference.obj')
    parser.add_argument('--dst_ref_obj', type=str, default='./data/head-poses/head-reference.obj')

    parser.add_argument('--src_marker_txt', type=str, default='./data/marker_txt_files/face_marker2.txt')
    parser.add_argument('--dst_marker_txt', type=str, default='./data/marker_txt_files/head_marker2.txt')

    parser.add_argument('--result_save_dir', type=str, default='./data/registration_result')
    parser.add_argument('--ph1_res_name', type=str, default='face_phase1_result.obj')
    parser.add_argument('--ph2_res_name', type=str, default='face_phase2_result.obj')

    parser.add_argument('--num_k', type=int, default=50)
    parser.add_argument('--thresh_dist', type=float, default=1.0)

    parser.add_argument('--ws', type=float, default=1.0)
    parser.add_argument('--wi', type=float, default=0.001)
    parser.add_argument('--wc', nargs="+", type=list, default=[1.0, 25.0, 45.0])
    return parser


def get_registration_solver_args():
    cfg_parser = get_registration_solver_config_parser()
    return cfg_parser.parse_args()


'''
Config parser of correspondence finder
'''


def get_correspondence_finder_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_reg_obj', type=str, default='./data/registration_result/face_phase2_result.obj')
    parser.add_argument('--dst_ref_obj', type=str, default='./data/head-poses/head-reference.obj')

    parser.add_argument('--num_k', type=int, default=50)
    parser.add_argument('--thresh_dist', type=float, default=0.2)

    parser.add_argument('--save_dir', type=str, default='./data/correspondence_result')
    parser.add_argument('--save_name', type=str, default='face_head_crspd.json')
    return parser


def get_correspondence_finder_args():
    cfg_parser = get_correspondence_finder_config_parser()
    return cfg_parser.parse_args()


'''
Config parser of deformation transfer solver
'''


def get_deformation_transfer_solver_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ref_obj', type=str, default='./data/face-poses/face-reference.obj')
    parser.add_argument('--dst_ref_obj', type=str, default='./data/head-poses/head-reference.obj')
    parser.add_argument('--src_def_obj', type=str, default='./data/face-poses/face-03-fury.obj')
    # if has multiple deformed source mesh to transfer
    parser.add_argument('--src_def_objs_dir', type=str, default='./data/face-poses')
    # path to correspondence json file
    parser.add_argument('--crspd_file', type=str, default='./data/correspondence_result/face_head_crspd.json')
    # whether source and target mesh has the same topology
    parser.add_argument('--same_topology', type=bool, default=False)

    parser.add_argument('--save_dir', type=str, default='./data/deformation_transfer_result')
    return parser


def get_deformation_transfer_solver_args():
    cfg_parser = get_deformation_transfer_solver_config_parser()
    return cfg_parser.parse_args()





