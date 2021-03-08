"""
    :author: Allan
    :copyright: © 2020 Yalun Hu <allancodeman@163.com>
    :license: MIT, see LICENSE for more details.
"""
import argparse


def get_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_k_1', type=int, default='./pretrained', help='directory for all savings')
    parser.add_argument('--threshold_distance_1', type=float, default='cuda:0', help='cuda device number, eg:[cuda:0]')
    parser.add_argument('--num_k_2', type=int, default='./pretrained', help='directory for all savings')
    parser.add_argument('--threshold_distance_2', type=float, default='cuda:0', help='cuda device number, eg:[cuda:0]')

    parser.add_argument('--', type=int, default=3, help='number of layers')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
    return parser


def get_args():
    cfg_parser = get_config_parser()
    return cfg_parser.parse_args()