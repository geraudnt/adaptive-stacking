#!/bin/bash

for i in `seq $1 $1`;
do 
 (
# Passive
python train.py --random_length --algo PPO --stack_type framestack --num_stack 8 --maze_length 6 --run $i
python train.py --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --run $i
python train.py --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 8 --maze_length 6 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i
python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 8 --maze_length 6 --run $i
python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --run $i
python train.py --arch transformer --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i

# Active
python train.py --random_length --active --algo PPO --stack_type framestack --num_stack 8 --maze_length 6 --run $i
python train.py --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --run $i
python train.py --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 8 --maze_length 6 --run $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --run $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type framestack --num_stack 8 --maze_length 6 --run $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --run $i
python train.py --arch transformer --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i

 ) >> log_PPO_active_$i &
done
