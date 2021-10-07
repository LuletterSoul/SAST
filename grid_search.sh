#!/usr/bin/env bash
layers=("r12" "r22" "r32" "r42")
for cl in ${layers[@]};
    do
	for sl in ${layers[@]};
	  do	 	
           CUDA_VISIBLE_DEVICES=3 python grid_search.py --content_dir /data/lxd/project/SAST/datasets/contents_1012 --style_dir /data/lxd/project/SAST/datasets/styles_1012 --kl 1 --cw 1 --sw 0 --lw 1 --cl $cl --sl $sl --update_step 50 --update_step_hr 50 --save_dir exp/2021-10-02_cw[1]_sw[0]_lw[1]_up[50] --gbp
          done
    done
