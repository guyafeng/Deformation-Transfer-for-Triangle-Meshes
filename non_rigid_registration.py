"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from config import get_registration_solver_args
from non_rigid_registration_solver import RegistrationSolver

"""
    this script will deform the source mesh into the shape of target mesh by applying non-rigid registration
    since we need to find the correspondence of two mesh with different topology
    It might take about 20 minutes to run the full pipe line, though it is kind of slow, we don't need to run it
    every-time. Once the deformed mesh is obtained and saved on disk, we could use it to calculate correspondence
    directly.
"""

if __name__ == "__main__":
    cfg = get_registration_solver_args()
    solver = RegistrationSolver(cfg)
    solver.non_rigid_registration()
