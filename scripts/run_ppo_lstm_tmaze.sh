#!/bin/bash
for i in `seq $1 $1`;
do 
 (

python train.py --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 0 --seed $i &
python train.py --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 2 --seed $i &
python train.py --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 6 --seed $i &
python train.py --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 14 --seed $i & 
python train.py --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 62 --seed $i &

python train.py --random_length --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 0 --seed $i &
python train.py --random_length --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 2 --seed $i &
python train.py --random_length --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 6 --seed $i &
python train.py --random_length --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 14 --seed $i & 
python train.py --random_length --arch mlp --algo RecurrentPPO --stack_type framestack --maze_length 62 --seed $i &

 ) >> log_RecurrentPPO_stacked_$i &
done
