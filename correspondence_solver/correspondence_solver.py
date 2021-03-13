"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
from .base_model import BasicCorresSolver


class CorresSolver(BasicCorresSolver):

    def __init__(self, cfg):
        super(CorresSolver, self).__init__(cfg=cfg)
        self.init_solver()

    def init_solver(self):
        pass