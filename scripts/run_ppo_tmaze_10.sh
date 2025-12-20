#!/bin/bash

for i in `seq $1 $1`;
do 
 (
# Active
python train.py --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 8 --run $i
python train.py --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 8 --run $i
python train.py --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 8 --run $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 8 --run $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 8 --run $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 8 --run $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 8 --run $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 8 --run $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 8 --run $i

 ) >> log_PPO_active_$i &
done
