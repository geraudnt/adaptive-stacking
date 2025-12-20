#!/bin/bash

for i in `seq $1 $1`;
do 
 (

python train.py --arch mlp --random_length --algo GRPO --stack_type framestack --num_stack 2 --maze_length 14 --run $i
python train.py --arch mlp --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 14 --run $i

python train.py --arch mlp --random_length --algo GRPO --stack_type framestack --num_stack 4 --maze_length 14 --run $i
python train.py --arch mlp --random_length --algo GRPO --stack_type adaptive --num_stack 4 --maze_length 14 --run $i

python train.py --arch mlp --random_length --algo GRPO --stack_type framestack --num_stack 8 --maze_length 14 --run $i
python train.py --arch mlp --random_length --algo GRPO --stack_type adaptive --num_stack 8 --maze_length 14 --run $i

python train.py --arch mlp --random_length --algo GRPO --stack_type framestack --num_stack 16 --maze_length 14 --run $i
python train.py --arch mlp --random_length --algo GRPO --stack_type adaptive --num_stack 16 --maze_length 14 --run $i

python train.py --arch mlp --random_length --algo GRPO --stack_type framestack --num_stack 32 --maze_length 14 --run $i
python train.py --arch mlp --random_length --algo GRPO --stack_type adaptive --num_stack 32 --maze_length 14 --run $i

 ) >> log_GRPO_active_$i &
done
