#!/bin/sh
echo "start FB15K"
# bs1 = 50, bs2 = 5
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15K --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 10
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15K --score_rel True --model ComplEx --rank 500 --learning_rate 0.08 --batch_size 500 --lmbda 0.01 --w_rel 0.1 --max_epochs 50
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15K --score_rel True --model ComplEx --rank 1000 --learning_rate 0.08 --batch_size 500 --lmbda 0.01 --w_rel 0.1 --max_epochs 50
CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15K --score_rel True --model ComplEx --rank 1500 --learning_rate 0.08 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 50
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset FB15K --score_rel True --model TransE --rank 1500 --learning_rate 0.08 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 50
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset FB15K --score_rel True --model RESCAL --rank 1500 --learning_rate 0.08 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 50
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset FB15K --model DistMult --rank 1500 --learning_rate 0.08 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 50
echo "finish FB15K"

# echo "start FB15K-237"
# bs1 = 1, bs2 = 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15K-237 --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15K-237 --score_rel True --model ComplEx --rank 1500 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 150
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset FB15K-237 --score_rel True --model TransE --rank 1500 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 150
# echo "finish FB15K-237"

# echo "start NELL"
# python main.py --dataset NELL995 --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 0 --max_epochs 100
# python main.py --dataset NELL995 --score_rel True --model ComplEx --rank 1500 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 0 --max_epochs 100
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset NELL995 --score_rel True --model TransE --rank 1500 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 0 --max_epochs 100
# echo "finish NELL"

# Define your hyperparameters
# ranks=('500' '800' '1000')
# learning_rates=('0.06' '0.08' '0.1')
# batch_sizes=('500' '800' '1000')

# # Loop through each combination of hyperparameters
# for r in "${ranks[@]}"; do
#     for lr in "${learning_rates[@]}"; do
#         for bs in "${batch_sizes[@]}"; do
#             CUDA_VISIBLE_DEVICES=0 python main.py \
#                 --dataset FB15K \
#                 --score_rel True \
#                 --model ComplEx \
#                 --rank "$r" \
#                 --learning_rate "$lr" \
#                 --batch_size "$bs" \
#                 --lmbda 0.01 \
#                 --w_rel 0.1 \
#                 --max_epochs 50
#         done
#     done
# done