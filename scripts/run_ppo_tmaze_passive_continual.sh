#!/bin/bash
for i in `seq $1 $1`;
do 
 (

python train.py --continual --algo PPO --stack_type framestack --num_stack 2 --maze_length 0 --seed $i
python train.py --continual --algo PPO --stack_type adaptive --num_stack 2 --maze_length 0 --seed $i

python train.py --continual --algo PPO --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --continual --algo PPO --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --continual --algo PPO --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i

python train.py --continual --algo PPO --stack_type framestack --num_stack 8 --maze_length 6 --seed $i
python train.py --continual --algo PPO --stack_type framestack --num_stack 2 --maze_length 6 --seed $i
python train.py --continual --algo PPO --stack_type adaptive --num_stack 2 --maze_length 6 --seed $i

python train.py --continual --algo PPO --stack_type framestack --num_stack 16 --maze_length 14 --seed $i
python train.py --continual --algo PPO --stack_type framestack --num_stack 2 --maze_length 14 --seed $i
python train.py --continual --algo PPO --stack_type adaptive --num_stack 2 --maze_length 14 --seed $i

python train.py --continual --algo PPO --stack_type framestack --num_stack 32 --maze_length 30 --seed $i
python train.py --continual --algo PPO --stack_type framestack --num_stack 2 --maze_length 30 --seed $i
python train.py --continual --algo PPO --stack_type adaptive --num_stack 2 --maze_length 30 --seed $i

 ) >> log_ppo_passive_$i &
done
