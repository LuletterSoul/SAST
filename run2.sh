#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python sast_transfer_convergency.py --content_dir /data/lxd/datasets/2021-10-09-weight_grid_search2/contents --style_dir /data/lxd/datasets/2021-10-08-weight_grid_search/styles --kl 1 --cw 1 --sw 0 --lw 1 --update_step 1 --update_step_hr 1 --save_dir exp/0425_cw1_lw30_k50_l1_4
