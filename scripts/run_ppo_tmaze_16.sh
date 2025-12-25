#!/bin/bash

for i in `seq $1 $1`;
do 
 (
# Passive
python train.py --random_length --algo PPO --stack_type framestack --num_stack 16 --maze_length 14 --seed $i
python train.py --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 14 --seed $i
python train.py --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 14 --seed $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 16 --maze_length 14 --seed $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 14 --seed $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 14 --seed $i
python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 16 --maze_length 14 --seed $i
python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 14 --seed $i
python train.py --arch transformer --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 14 --seed $i

 ) >> log_PPO_active_$i &
done
