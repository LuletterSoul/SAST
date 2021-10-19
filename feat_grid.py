#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: cmp.py
@time: 4/23/20 9:40 AM
@version 1.0
@desc:
"""
import cv2
import os
import numpy as np
from pathlib import Path

from numpy.testing._private.utils import print_assert_equal

content_dir = '/data/lxd/project/SAST/datasets/contents_1012'
style_dir = '/data/lxd/project/SAST/datasets/styles_1012'
# style_dir = 'images/styles_0421_k'
# content_dir = 'images/contents_0421_k'
# stylizations = ['exp/0421_cw1_lw0.1_kl50', 'exp/0421_cw1_lw1_kl50', 'exp/0421_cw1_lw10_kl50', 'exp/0421_cw1_lw100_kl50',
#                 'exp/0421_cw1_lw1000_kl50', 'exp/0421_cw1_lw10000_kl50', 'exp/0421_cw1_lw100000_kl50']

# stylizations = ['exp/0421_cw1_lw10_kl50', 'exp/0421_cw1_lw20_kl50', 'exp/0421_cw1_lw30_kl50', 'exp/0421_cw1_lw40_kl50',
#                 'exp/0421_cw1_lw50_kl50', 'exp/0421_cw1_lw60_kl50', 'exp/0421_cw1_lw70_kl50', 'exp/0421_cw1_lw80_kl50',
#                 'exp/0421_cw1_lw90_kl50', 'exp/0421_cw1_lw100_kl50']
# stylizations = [['exp/0428_cw1_lw30_k1', 'exp/0428_cw1_lw30_k5', 'exp/0428_cw1_lw30_k10', 'exp/0428_cw1_lw30_k50',
#                  'exp/0428_cw1_lw30_k100', 'exp/0428_cw1_lw30_k500', 'exp/0428_cw1_lw30_k1000'],
#                 ['exp/0423_cw1_lw50_k1', 'exp/0423_cw1_lw50_k5', 'exp/0423_cw1_lw50_k10', 'exp/0423_cw1_lw50_k50',
#                  'exp/0423_cw1_lw50_k100', 'exp/0423_cw1_lw50_k500', 'exp/0423_cw1_lw50_k1000']]
# stylizations = [['exp/0428_cw1_lw30_nups', 'exp/0428_cw1_lw30_ups200', 'exp/0428_cw1_lw30_ups100',
#                  'exp/0428_cw1_lw30_ups50', 'exp/0428_cw1_lw30_ups40',
#                  'exp/0428_cw1_lw30_ups30',
#                  'exp/0428_cw1_lw30_ups20', 'exp/0428_cw1_lw30_ups10', 'exp/0428_cw1_lw30_ups1']]
# stylizations = [
#     ['exp/0428_cw1_lw0.1_nups', 'exp/0428_cw1_lw1_nups', 'exp/0428_cw1_lw10_nups', 'exp/0428_cw1_lw100_nups',
#      'exp/0428_cw1_lw1000_nups', 'exp/0428_cw1_lw10000_nups', 'exp/0428_cw1_lw10000_nups']]

# stylizations = [
#     ['exp/0429_cw0_lw1_nups', 'exp/0429_cw0_lw1_50ups', 'exp/0429_cw0_lw1_40ups', 'exp/0429_cw0_lw1_30ups'
#         , 'exp/0429_cw0_lw1_20ups'
#         , 'exp/0429_cw0_lw1_10ups']]
# stylizations = [
#     ['exp/0429_cw0_lw1','exp/0429_cw0_lw1000']]
# stylizations = [[
#     'exp/0428_cw1_lw0.1', 'exp/0428_cw1_lw0.2', 'exp/0428_cw1_lw0.3',
#     'exp/0428_cw1_lw0.4', 'exp/0428_cw1_lw0.5', 'exp/0428_cw1_lw0.6',
#     'exp/0428_cw1_lw0.7', 'exp/0428_cw1_lw0.8', 'exp/0428_cw1_lw0.9',
#     'exp/0428_cw1_lw1_nups'
# ]]

# 固定风格特征r32层，内容特征分别使用r12-r42, cw = 0
# stylizations = [[
#     'exp/2021-10-01_cw1_sw0_lw1_us1000_ushr1000_cfm[r12]',
#     'exp/2021-10-01_cw1_sw0_lw1_us1000_ushr1000_cfm[r22]',
#     'exp/2021-10-01_cw1_sw0_lw1_us1000_ushr1000_cfm[r32]',
#     'exp/2021-10-01_cw1_sw0_lw1_us1000_ushr1000_cfm[r42]',
#     'exp/2021-10-01_cw0_sw0_lw1_us1000_ushr1000_cfm[r42]'
# ]]
# output_path = Path('output/2021-10-01_cw1_sw0_lw1_us1000_ushr1000_cfm')

# lw = 0， 内容特征分别使用r12-r51
# stylizations = [[
#     'exp/2021-10-01_cw[1]_sw[0]_lw[0]_us1000_ushr1000_cfm[r12]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[0]_us1000_ushr1000_cfm[r22]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[0]_us1000_ushr1000_cfm[r32]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[0]_us1000_ushr1000_cfm[r42]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[0]_us1000_ushr1000_cfm[r52]',
# ]]
# output_path = Path('output/2021-10-01_cw[1]_sw[0]_lw[0]_us1000_ushr1000_cfm')

# # 固定内容特征r42层，风格特征分别使用r32-r52, cw = 0

content_rows = ['r12', 'r22', 'r32', 'r42', 'r52']
style_cols = ['r12', 'r22', 'r32', 'r42', 'r52']

# stylizations = [[
#     'exp/2021-10-01_cw[1]_sw[0]_lw[1]_us1000_ushr1000_sfm[r12]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[1]_us1000_ushr1000_sfm[r22]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[1]_us1000_ushr1000_sfm[r32]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[1]_us1000_ushr1000_sfm[r42]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[1]_us1000_ushr1000_sfm[r52]',
# ]]
output_dir = Path('output/2021-10-07_features_gridsearch_noups')

# # 固定风格特征r42层, 内容损失权重分别为1,2,5,10
# stylizations = [[
#     'exp/2021-10-01_cw[1]_sw[0]_lw[1]_us1000_ushr1000_sfm[r42]',
#     'exp/2021-10-01_cw[2]_sw[0]_lw[1]_us1000_ushr1000_sfm[r42]',
#     'exp/2021-10-01_cw[5]_sw[0]_lw[1]_us1000_ushr1000_sfm[r42]',
#     'exp/2021-10-01_cw[10]_sw[0]_lw[1]_us1000_ushr1000_sfm[r42]',
#     'exp/2021-10-01_cw[1]_sw[0]_lw[1]_us1000_ushr1000_sfm[r32]'
# ]]
# output_path = Path('output/2021-10-01_cw[1,2,5,10]_sw[0]_lw[1]_us1000_ushr1000_sfm[r42]')

# stylizations = [
# ['exp/0428_cw1_lw0.001','exp/0428_cw1_lw0.01','exp/0428_cw1_lw0.1', 'exp/0428_cw1_lw0.3', 'exp/0428_cw1_lw0.5',
#  'exp/0428_cw1_lw0.7', 'exp/0428_cw1_lw0.9', 'exp/0428_cw1_lw1_nups']]

# stylizations = ['exp/0427_cw1_lw30_k50_ups50_it500',
#                 'exp/0427_cw1_lw30_k50_ups50_it400', 'exp/0427_cw1_lw30_k50_ups50_it300', 'exp/0427_cw1_lw30_k50_ups50_it200',
#                 'exp/0427_cw1_lw30_k50_ups50_it100', 'exp/0427_cw1_lw30_k50_ups50_it50', 'exp/0427_cw1_lw30_k50_ups50_it40',
#                 'exp/0427_cw1_lw30_k50_ups50_it30', 'exp/0427_cw1_lw30_k50_ups50_it20']
# output_path = Path('output/contents_0427_it_cmp')
# output_path = Path('output/contents_0429_cw0_lw1-1000_cmp')
# output_path = Path('outpu1/contents_0429_cw0_lw1_ups_cmp')

if not output_dir.exists():
    output_dir.mkdir(exist_ok=True, parents=True)

style_names = os.listdir(style_dir)
content_names = os.listdir(content_dir)

WHITE = np.ones((512, 512, 3), dtype=np.uint8) * 255

for c in content_names:
    for s in style_names:
        plot = []
        c_idx, extention = os.path.splitext(c)
        c_idx = c_idx.replace(' ', '_')
        content_img = cv2.imread(os.path.join(content_dir, c))
        h_plot = [content_img]
        for i in range(len(style_cols) + 1):
            h_plot.append(WHITE)
        plot.append(np.hstack(h_plot))
        s_idx, extention = os.path.splitext(s)
        style_img = cv2.imread(os.path.join(style_dir, s))
        for cl in content_rows:
            h_plot = [WHITE]
            for sl in style_cols:
                # o_path = f'exp/2021-10-02_cw[1]_sw[0]_lw[1]_cl[{cl}]_sl[{sl}]'
                # o_path = f'exp/2021-10-02_cw[1]_sw[0]_lw[1]_up[50]_cl[{cl}]_sl[{sl}]'
                # o_path = f'exp/2021-10-02_cw[1]_sw[0]_lw[1]_up[50]_cl[{cl}]_sl[{sl}]'
                o_path = f'exp/2021-10-07_up[1000]_cw[1]_lw[1]_cl[{cl}]_sl[{sl}]'
                # for sty in stylizations:
                # s_idx, extention = os.path.splitext(s)
                print(os.path.join(o_path, c_idx, f'{c_idx}-{s_idx}.png'))
                output = cv2.imread(
                    os.path.join(o_path, c_idx, f'{c_idx}-{s_idx}.png'))
                if output is None:
                    raise Exception('Empty stylization loaded.')
                h_plot.append(output)
            h_plot.append(WHITE)
            plot.append(np.hstack(h_plot))
        h_plot = []
        for i in range(len(style_cols) + 1):
            h_plot.append(WHITE)
        h_plot.append(style_img)
        plot.append(np.hstack(h_plot))
        plot = np.vstack(plot)
        output_path = f'{str(output_dir)}/{c_idx}-{s_idx}.png'
        cv2.imwrite(output_path, plot)
