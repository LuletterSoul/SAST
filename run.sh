#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py --content_dir /data/lxd/project/SAST/datasets/contents_1012 --style_dir /data/lxd/project/SAST/datasets/styles_1012 --kl 1 --cw 1 --sw 0 --lw 1 --save_dir exp/2021-10-02_cw[1]_sw[0]_lw[1] --gbp
