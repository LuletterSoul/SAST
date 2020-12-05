# Semantically aligned style transfer

## Run command example

    CUDA_VISIBLE_DEVICES=0 python sast_transfer.py \
    --content_dir [content image directory] \
    --style_dir [style image directory] \
    --kl [k nearest number of affinity matrix]
    --cw [content loss weight] \
    --sw [style loss weight] \
    --lw [laplacian loss weight] \
    --update_step [updating frequency of affinity matrix on first stage] \
    --update_step_hr [updating frequency of affinity matrix on second stage] \
    --save_dir [output directory]
