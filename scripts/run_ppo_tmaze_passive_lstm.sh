#!/bin/bash

# trainort CUDA_VISIBLE_DEVICES=""

for i in `seq $1 $1`;
do 
 (
# python train.py --arch lstm --random_length --algo PPO --stack_type no_stack --maze_length 0 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 64 --maze_length 0 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 0 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 0 --run $i

# python train.py --arch lstm --random_length --algo PPO --stack_type no_stack --maze_length 2 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 64 --maze_length 2 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --run $i

# python train.py --arch lstm --random_length --algo PPO --stack_type no_stack --maze_length 6 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 64 --maze_length 6 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i

# python train.py --arch lstm --random_length --algo PPO --stack_type no_stack --maze_length 14 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 64 --maze_length 14 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 14 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 14 --run $i

# python train.py --arch lstm --random_length --algo PPO --stack_type no_stack --maze_length 62 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 64 --maze_length 62 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type framestack --num_stack 2 --maze_length 62 --run $i
python train.py --arch lstm --random_length --algo PPO --stack_type adaptive --num_stack 2 --maze_length 62 --run $i

 ) >> log_PPO_active_$i &
done
