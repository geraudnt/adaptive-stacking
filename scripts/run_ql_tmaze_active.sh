#!/bin/bash
for i in `seq 0 19`;
do 
 (

python train.py --active --algo QL --stack_type no_stack --maze_length 0 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 0 --seed $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 0 --seed $i

python train.py --active --algo QL --stack_type no_stack --maze_length 1 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 1 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 3 --maze_length 1 --seed $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 1 --seed $i

python train.py --active --algo QL --stack_type no_stack --maze_length 2 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 2 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 4 --maze_length 2 --seed $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 2 --seed $i

python train.py --active --algo QL --stack_type no_stack --maze_length 3 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 3 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 5 --maze_length 3 --seed $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 3 --seed $i

python train.py --active --algo QL --stack_type no_stack --maze_length 4 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 6 --maze_length 4 --seed $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 4 --seed $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 4 --seed $i

 ) >> log_ql_active_$i &
done
