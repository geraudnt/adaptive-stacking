#!/bin/bash
for i in `seq 0 4`;
do 
 (

python train.py --env xormaze-v0 --algo QL --stack_type demir --num_stack 3 --maze_length 1 --seed $i &
python train.py --env xormaze-v0 --algo QL --stack_type demir --num_stack 3 --maze_length 2 --seed $i &
python train.py --env xormaze-v0 --algo QL --stack_type demir --num_stack 3 --maze_length 1 --intrinsic_rewards --seed $i &
python train.py --env xormaze-v0 --algo QL --stack_type demir --num_stack 3 --maze_length 2 --intrinsic_rewards --seed $i &

# python train.py --env xormaze-v0 --algo QL --stack_type framestack --num_stack 10 --maze_length 2 --seed $i &
# python train.py --env xormaze-v0 --algo QL --stack_type framestack --num_stack 3 --maze_length 2 --seed $i &
# python train.py --env xormaze-v0 --algo QL --stack_type masked --num_stack 10 --maze_length 2 --seed $i &
# python train.py --env xormaze-v0 --algo QL --stack_type masked --num_stack 3 --maze_length 2 --seed $i &

 ) >> log_ql_xormaze_$i &
done
