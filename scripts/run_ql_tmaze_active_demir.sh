#!/bin/bash
for i in `seq $1 $1`;
do 
 (

python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 0 --run $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 1 --run $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 2 --run $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 3 --run $i
python train.py --active --algo QL --stack_type demir --num_stack 2 --maze_length 4 --run $i

python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 0 --run $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 1 --run $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 2 --run $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 3 --run $i
python train.py --active --intrinsic_rewards --algo QL --stack_type demir --num_stack 2 --maze_length 4 --run $i

 ) >> log_ql_active_$i &
done
