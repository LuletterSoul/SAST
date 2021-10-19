#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py --content_dir /data/lxd/datasets/2021-10-09-weight_grid_search2/contents --style_dir  /data/lxd/datasets/2021-10-08-weight_grid_search/styles  --kl 1 --cw 1 --sw 0 --lw 30 --save_dir exp/2021-10-19_test --update_step 10 --update_step_hr 10 --gbp
