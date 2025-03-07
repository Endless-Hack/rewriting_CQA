#!/bin/sh
echo "start FB15K"
CUDA_VISIBLE_DEVICES=1 python main.py --data_path data/FB15K-betae --kbc_path kbc/FB15K/best_valid.model --fraction 10 --thrshd 0.001 --neg_scale 6 
--query_rew 

# echo "start FB15K-237"
# CUDA_VISIBLE_DEVICES=1 python main.py --data_path data/FB15K-237-betae --kbc_path kbc/FB15K-237/best_valid.model --fraction 10 --thrshd 0.0002 --neg_scale 3 \
# --query_rew

# echo "start NELL"
# python main.py --data_path data/NELL-betae --kbc_path kbc/NELL995/best_valid.model --fraction 10 --thrshd 0.0002 --neg_scale 6 \
# --query_rew

echo "Finish"

