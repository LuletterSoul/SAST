#!/usr/bin/env bash
# layers=("r12" "r22" "r32" "r42")
c_layers=("r12" "r22" "r32" "r42" "r52")
# c_layers=("r52")
# s_layers=("r12" "r22" "r32" "r42")
s_layers=("r52")
for cl in ${c_layers[@]};
    do
	for sl in ${s_layers[@]};
	  do	 	
           CUDA_VISIBLE_DEVICES=0 python grid_search.py --content_dir /data/lxd/project/SAST/datasets/contents_1012 --style_dir /data/lxd/project/SAST/datasets/styles_1012 --kl 1 --cw 1 --sw 0 --lw 1 --cl $cl --sl $sl --update_step 1000 --update_step_hr 1000 --save_dir exp/2021-10-07_up[1000]_cw[1]_lw[1] --gbp
          done
    done
