#!/bin/bash
for i in `seq $1 $1`;
do 
 (

python train.py --env cube-v0 --algo GRPO --stack_type framestack --num_stack 10 --cube_cam face --scramble_steps 5 --maxiter 10000000 --seed $i &
python train.py --env cube-v0 --algo GRPO --stack_type adaptive --num_stack 10 --cube_cam face --scramble_steps 5 --maxiter 10000000 --seed $i &

 ) >> log_grpo_$i
done
