#!/usr/bin/env bash
# cws=(1 10 50 100 1000)
cws=(1)
# lws=(0.1 1 10 100 1000)
lws=(1)
# content_dir=/data/lxd/datasets/2021-10-08-weight_grid_search/contents
content_dir=/data/lxd/datasets/2021-10-09-weight_grid_search2/contents
style_dir=/data/lxd/datasets/2021-10-08-weight_grid_search/styles
for cw in ${cws[@]};
    do
	for lw in ${lws[@]};
	  do	 	
          CUDA_VISIBLE_DEVICES=2 python exp_eval_convergency.py --content_dir $content_dir --style_dir $style_dir --kl 1 --sw 0 --cw $cw --lw $lw --cl r42 --sl r32 --update_step 10 --update_step_hr 10 --save_dir exp/2021-10-19_cl[r42]_sl[r32]_up[1]_convergency_2 --gbp
          done
    done
