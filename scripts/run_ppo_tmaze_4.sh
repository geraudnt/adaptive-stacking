#!/bin/bash

trainort CUDA_VISIBLE_DEVICES=""

for i in `seq $1 $1`;
do 
 (
# Passive
python train.py --random_length --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i
python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --arch transformer --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i

# Active
python train.py --random_length --active --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i

 ) >> log_PPO_active_$i &
done
