#!/bin/bash
for i in `seq 0 4`;
do 
 (

python train.py --env xormaze-v0 --algo QL --stack_type framestack --num_stack 5 --maze_length 1 --run $i &
python train.py --env xormaze-v0 --algo QL --stack_type framestack --num_stack 3 --maze_length 1 --run $i &
python train.py --env xormaze-v0 --algo QL --stack_type adaptive --num_stack 5 --maze_length 1 --run $i &
python train.py --env xormaze-v0 --algo QL --stack_type adaptive --num_stack 3 --maze_length 1 --run $i &

# python train.py --env xormaze-v0 --algo QL --stack_type framestack --num_stack 10 --maze_length 2 --run $i &
# python train.py --env xormaze-v0 --algo QL --stack_type framestack --num_stack 3 --maze_length 2 --run $i &
# python train.py --env xormaze-v0 --algo QL --stack_type adaptive --num_stack 10 --maze_length 2 --run $i &
# python train.py --env xormaze-v0 --algo QL --stack_type adaptive --num_stack 3 --maze_length 2 --run $i &

 ) >> log_ql_xormaze_$i &
done
