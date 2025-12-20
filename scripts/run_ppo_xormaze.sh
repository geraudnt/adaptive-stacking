#!/bin/bash
for i in `seq $1 $1`;
do 
 (

python train.py --env xormaze-v0 --algo PPO --stack_type framestack --num_stack 10 --maze_length 2  --maxiter 10000000 --run $i 
python train.py --env xormaze-v0 --algo PPO --stack_type adaptive --num_stack 10 --maze_length 2 --maxiter 10000000 --run $i

 ) >> log_ppo_xormaze_$i &
done
