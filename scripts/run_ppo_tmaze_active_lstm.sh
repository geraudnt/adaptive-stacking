#!/bin/bash

# trainort CUDA_VISIBLE_DEVICES=""

for i in `seq $1 $1`;
do 
 (
# python train.py --arch lstm --random_length --active --algo PPO --stack_type no_stack --maze_length 0 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 0 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 0 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 0 --seed $i

# python train.py --arch lstm --random_length --active --algo PPO --stack_type no_stack --maze_length 2 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i

# python train.py --arch lstm --random_length --active --algo PPO --stack_type no_stack --maze_length 4 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 4 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 4 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 4 --seed $i

# python train.py --arch lstm --random_length --active --algo PPO --stack_type no_stack --maze_length 6 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 6 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --seed $i

# python train.py --arch lstm --random_length --active --algo PPO --stack_type no_stack --maze_length 8 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 10 --maze_length 8 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type framestack --num_stack 2 --maze_length 8 --seed $i
python train.py --arch lstm --random_length --active --algo PPO --stack_type adaptive --num_stack 2 --maze_length 8 --seed $i

 ) >> log_PPO_active_$i &
done
