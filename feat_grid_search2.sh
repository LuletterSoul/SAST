#!/usr/bin/env bash
# layers=("r12" "r22" "r32" "r42")
c_layers=("r42")
# c_layers=("r52")
# s_layers=("r12" "r22" "r32" "r42")
s_layers=("r12" "r22" "r32" "r42" "r52")
for cl in ${c_layers[@]};
    do
	for sl in ${s_layers[@]};
	  do	 	
           CUDA_VISIBLE_DEVICES=1 python grid_search.py --content_dir /data/lxd/datasets/2021-10-09-weight_grid_search3/contents --style_dir /data/lxd/datasets/2021-10-09-weight_grid_search3/styles --kl 1 --cw 1 --sw 0 --lw 1 --cl $cl --sl $sl --update_step 10000 --update_step_hr 10000 --save_dir exp/2021-10-19_test_time --gbp
          done
    done
