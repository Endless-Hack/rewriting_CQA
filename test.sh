#!/bin/sh
echo "start FB15k"
CUDA_VISIBLE_DEVICES=1 python main.py --data_path data/FB15k-betae --kbc_path kbc/FB15K/best_valid.model --fraction 10 --thrshd 0.001 --neg_scale 6 \
#--query_rew
echo "Finish"

