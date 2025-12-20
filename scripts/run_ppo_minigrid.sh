#!/bin/bash

(
# python train.py --arch cnn --random_length --algo RecurrentPPO --stack_type framestack --num_stack 1 --seed 0 --env MiniGrid-MemoryS13-v0 --n_steps 1024 --run $1 &
# python train.py --arch lstm --with_cnn --random_length --algo PPO --stack_type framestack --num_stack 2 --seed 0 --env MiniGrid-MemoryS13-v0 --n_steps 1024 --run $1 &
# python train.py --arch lstm --with_cnn --random_length --algo PPO --stack_type adaptive --num_stack 2 --seed 0 --env MiniGrid-MemoryS13-v0 --n_steps 1024 --run $1 &

python train.py --arch mlp --with_cnn --algo PPO --stack_type framestack --num_stack 2 --env MiniGrid-MemoryS17Random-v0 --n_envs 8 --features_dim 1024 --hidden_size 1024 --run $1
python train.py --arch mlp --with_cnn --algo PPO --stack_type adaptive --num_stack 2 --env MiniGrid-MemoryS17Random-v0 --n_envs 8 --features_dim 1024 --hidden_size 1024 --run $1
python train.py --arch mlp --with_cnn --algo PPO --stack_type framestack --num_stack 17 --env MiniGrid-MemoryS17Random-v0 --n_envs 8 --features_dim 1024 --hidden_size 1024 --run $1
python train.py --arch mlp --with_cnn --algo PPO --stack_type adaptive --num_stack 17 --env MiniGrid-MemoryS17Random-v0 --n_envs 8 --features_dim 1024 --hidden_size 1024 --run $1
python train.py --arch cnn --algo RecurrentPPO --stack_type framestack --env MiniGrid-MemoryS17Random-v0 --n_envs 8 --features_dim 1024 --hidden_size 1024 --run $1

) >> log_PPO_minigrid_$1 &
