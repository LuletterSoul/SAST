#!/usr/bin/env bash
style_dir=/data/lxd/datasets/2021-10-08-pggan-selected
content_dir=/data/lxd/datasets/2021-10-08-warped/4
CUDA_VISIBLE_DEVICES=3 python main.py --content_dir $content_dir --style_dir $style_dir --kl 1 --sw 0 --cw 1 --lw 1 --cl r42 --sl r32 --update_step 50 --update_step_hr 50 --save_dir exp/2021-10-08_cw[1]_lw[1]_cl[r42]_sl[r32]_up[50] --gbp
