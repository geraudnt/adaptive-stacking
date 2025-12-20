#!/bin/bash

for i in `seq $1 $1`;
do 
 (
python train.py --arch mlp --random_length --algo PPO --stack_type framestack --num_stack 1000 --maze_length 998 --run $i &
python train.py --arch mlp --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 998 --run $i &
python train.py --arch mlp --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 998 --run $i &

python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 1000 --maze_length 998 --run $i &
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 998 --run $i &
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 998 --run $i &

python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 1000 --maze_length 998 --run $i &
python train.py --arch transformer --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 998 --run $i &
python train.py --arch transformer --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 998 --run $i &


 ) >> log_PPO_passive_1000_$i &
done
