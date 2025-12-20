#!/bin/bash

(
if [[ "$1" -eq 1 ]]; then
python train.py --env cube-v0 --random_length --algo PPO --maxiter $2 --arch $3 --stack_type framestack --num_stack $4 --cube_cam $5 --scramble_steps $6 --run $7
python train.py --env cube-v0 --random_length --algo PPO --maxiter $2 --arch $3 --stack_type adaptive --num_stack $4 --cube_cam $5 --scramble_steps $6 --run $7
else
python train.py --env cube-v0 --algo PPO --maxiter $2 --arch $3 --stack_type framestack --num_stack $4 --cube_cam $5 --scramble_steps $6 --run $7
python train.py --env cube-v0 --algo PPO --maxiter $2 --arch $3 --stack_type adaptive --num_stack $4 --cube_cam $5 --scramble_steps $6 --run $7
fi
) >> log_PPO_cube_$7 &
