#!/bin/bash
for i in `seq 0 19`;
do 
 (

python train.py --active --algo QL --stack_type no_stack --maze_length 0 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 0 --run $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 0 --run $i

python train.py --active --algo QL --stack_type no_stack --maze_length 1 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 1 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 3 --maze_length 1 --run $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 1 --run $i

python train.py --active --algo QL --stack_type no_stack --maze_length 2 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 2 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 4 --maze_length 2 --run $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 2 --run $i

python train.py --active --algo QL --stack_type no_stack --maze_length 3 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 3 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 5 --maze_length 3 --run $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 3 --run $i

python train.py --active --algo QL --stack_type no_stack --maze_length 4 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 6 --maze_length 4 --run $i
python train.py --active --algo QL --stack_type framestack --num_stack 2 --maze_length 4 --run $i
python train.py --active --algo QL --stack_type adaptive --num_stack 2 --maze_length 4 --run $i

 ) >> log_ql_active_$i &
done
