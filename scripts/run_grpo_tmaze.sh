#!/bin/bash
for i in `seq $1 $1`;
do 
 (

python train.py --env popgym-PositionOnlyCartPoleHard --algo GRPO --stack_type adaptive --num_stack 2 --run $i
python train.py --env popgym-VelocityOnlyCartPoleHard --algo GRPO --stack_type adaptive --num_stack 2 --run $i
python train.py --env cube-v0 --algo GRPO --stack_type adaptive --num_stack 10 --cube_cam face --scramble_steps 5 --run $i

python train.py --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 0 --run $i
python train.py --arch lstm --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 0 --run $i
python train.py --arch transformer --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 0 --run $i

python train.py --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 2 --run $i
python train.py --arch lstm --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 2 --run $i
python train.py --arch transformer --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 2 --run $i

python train.py --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i
python train.py --arch lstm --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i
python train.py --arch transformer --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 6 --run $i

python train.py --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 14 --run $i
python train.py --arch lstm --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 14 --run $i
python train.py --arch transformer --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 14 --run $i

python train.py --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 30 --run $i
python train.py --arch lstm --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 30 --run $i
python train.py --arch transformer --random_length --algo GRPO --stack_type adaptive --num_stack 2 --maze_length 30 --run $i

 ) >> log_GRPO_adaptive_$i &
done
