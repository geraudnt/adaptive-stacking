#!/bin/bash

(
python train.py --env popgym-PositionOnlyCartPoleHard --algo PPO --arch $1 --stack_type framestack --num_stack $2 --seed $3
python train.py --env popgym-PositionOnlyCartPoleHard --algo PPO --arch $1 --stack_type adaptive --num_stack $2 --seed $3
python train.py --env popgym-VelocityOnlyCartPoleHard --algo PPO --arch $1 --stack_type framestack --num_stack $2 --seed $3
python train.py --env popgym-VelocityOnlyCartPoleHard --algo PPO --arch $1 --stack_type adaptive --num_stack $2 --seed $3
python train.py --env popgym-NoisyPositionOnlyCartPole --algo PPO --arch $1 --stack_type framestack --num_stack $2 --seed $3
python train.py --env popgym-NoisyPositionOnlyCartPole --algo PPO --arch $1 --stack_type adaptive --num_stack $2 --seed $3

) >> log_PPO_popgym_$3 &
