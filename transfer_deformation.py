"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from deformation_transfer import DeformationTransferSolver
from config import get_deformation_transfer_solver_args
import process_obj_file as p_obj
import os

"""
    This script shows a demo that how to execute the deformation transfer algorithm
"""

if __name__ == "__main__":
    cfg = get_deformation_transfer_solver_args()
    solver = DeformationTransferSolver(cfg)

    src_def_vts, _ = p_obj.load_obj_file(cfg.src_def_obj)
    _, dst_ref_faces = p_obj.load_obj_file(cfg.dst_ref_obj)
    solver_right_mat = solver.build_problem(src_def_vts)
    dst_def_vts = solver.solve_problem(solver_right_mat)

    save_path = os.path.join(cfg.save_dir, "dt_result.obj")
    p_obj.write_obj_file(save_path, dst_def_vts, dst_ref_faces)


