#!/bin/bash

(
python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch mlp --stack_type framestack --num_stack 50 --seed $1 &
python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch mlp --stack_type framestack --num_stack 4 --seed $1 &
python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch mlp --stack_type adaptive --num_stack 4 --seed $1 &

python train.py --env FetchReachDense-v4 --algo RecurrentPPO --n_steps 1024 --batch_size 1024 --single_head --arch mlp --stack_type framestack --num_stack 1 --seed $1 &
python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch lstm --stack_type framestack --num_stack 4 --seed $1 &
python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch lstm --stack_type adaptive --num_stack 4 --seed $1 &

python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch transformer --stack_type framestack --num_stack 50 --seed $1 &
python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch transformer --stack_type framestack --num_stack 4 --seed $1 &
python train.py --env FetchReachDense-v4 --algo PPO --n_steps 1024 --batch_size 1024 --single_head --arch transformer --stack_type adaptive --num_stack 4 --seed $1 &

) >> log_PPO_fetch_$1 &
