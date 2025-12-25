#!/bin/bash
for i in `seq $1 $1`;
do 
 (

python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 0 --seed $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 1 --seed $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 2 --seed $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 3 --seed $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 4 --seed $i

python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 0 --seed $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 1 --seed $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 2 --seed $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 3 --seed $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 4 --seed $i

 ) >> log_ql_active_$i &
done
