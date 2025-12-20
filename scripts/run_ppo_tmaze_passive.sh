#!/bin/bash

trainort CUDA_VISIBLE_DEVICES=""

for i in `seq $1 $1`;
do 
 (
python train.py --active --algo PPO --stack_type no_stack --maze_length 0 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 0 --run $i
python train.py --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 0 --run $i

python train.py --active --algo PPO --stack_type no_stack --maze_length 1 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 3 --maze_length 1 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 1 --run $i
python train.py --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 1 --run $i

python train.py --active --algo PPO --stack_type no_stack --maze_length 2 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --run $i
python train.py --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --run $i

python train.py --active --algo PPO --stack_type no_stack --maze_length 3 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 3 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 5 --maze_length 3 --run $i
python train.py --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 3 --run $i

python train.py --active --algo PPO --stack_type no_stack --maze_length 4 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 6 --maze_length 4 --run $i
python train.py --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 4 --run $i
python train.py --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 4 --run $i

 ) >> log_PPO_passive_$i &
done
