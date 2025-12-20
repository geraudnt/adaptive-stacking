import gymnasium as gym
from envs.wrappers import *


# Environment setup (unchanged)
def make_env(args):
    name = f"{args.algo}-arch_{args.arch}-env_{args.env}-num_stack_{args.num_stack}-stack_type_{args.stack_type}-run_{args.run}"

    if "tmaze" in args.env: # tmaze-v0
        import envs.GridWorld

        env = gym.make(args.env, length=args.maze_length, random_length=args.random_length, active=args.active,
                    continual=args.no_reset, fix_start=True, goal_obs=False,
                    fully_obs=args.fully_obs, render_mode=args.render_mode)
        maze_type = "active" if args.active else "passive"
        name = f"{args.algo}-arch_{args.arch}-env_{maze_type}_tmaze-v0_maze_length_{args.maze_length+2}-random_length_{args.random_length}-num_stack_{args.num_stack}-stack_type_{args.stack_type}-run_{args.run}"
        if args.no_reset:
            name = f"{args.algo}-arch_{args.arch}-env_{maze_type}_tmaze-continual-v0_maze_length_{args.maze_length+2}-random_length_{args.random_length}-num_stack_{args.num_stack}-stack_type_{args.stack_type}-run_{args.run}"
    
    elif "xormaze" in args.env: # xormaze-v0
        import envs.GridWorldXOR

        env = gym.make(args.env, length=args.maze_length, random_length=args.random_length, active=args.active,
                    continual=args.no_reset, fix_start=True, goal_obs=False,
                    fully_obs=args.fully_obs, render_mode=args.render_mode)
        maze_type = "active" if args.active else "passive"
        name = f"{args.algo}-arch_{args.arch}-env_{maze_type}_xormaze-v0_maze_length_{args.maze_length}-random_length_{args.random_length}-num_stack_{args.num_stack}-stack_type_{args.stack_type}-run_{args.run}"
        if args.no_reset:
            name = f"{args.algo}-arch_{args.arch}-env_{maze_type}_xormaze-continual-v0_maze_length_{args.maze_length}-random_length_{args.random_length}-num_stack_{args.num_stack}-stack_type_{args.stack_type}-run_{args.run}"
    
    elif "cube" in args.env: # cube-v0
        import envs.cube2x2

        if args.fully_obs:
            cube_cam = "full"
        else: 
            cube_cam = args.cube_cam
        env = gym.make(args.env, episode_steps=100, scramble_steps=args.scramble_steps, random_length=args.random_length,
                    cube_cam=cube_cam, render_mode=args.render_mode, render_2d=args.render_2d)
        name = f"{args.algo}-arch_{args.arch}-env_cube-v0_scramble_steps_{args.scramble_steps}-random_length_{args.random_length}-cube_cam_{cube_cam}-num_stack_{args.num_stack}-stack_type_{args.stack_type}-run_{args.run}"
    
    elif "popgym" in args.env: # E.g. popgym-PositionOnlyCartPoleHard-v0
        import popgym
        from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction

        if "PositionOnlyCartPoleHard" in args.env:
            env = popgym.envs.position_only_cartpole.PositionOnlyCartPoleHard()
            env = DiscreteAction(Flatten(PreviousAction(env)))
        elif "VelocityOnlyCartPoleHard" in args.env:
            env = popgym.envs.velocity_only_cartpole.VelocityOnlyCartPoleHard()
            env = DiscreteAction(Flatten(PreviousAction(env))) 
        elif "NoisyPositionOnlyCartPole" in args.env:
            env = popgym.envs.noisy_position_only_cartpole.NoisyPositionOnlyCartPole()
            env = DiscreteAction(Flatten(PreviousAction(env))) 
        elif "concentration" in args.env:
            env = popgym.envs.concentration.ConcentrationEasy()
            env = DiscreteAction(Flatten(PreviousAction(env)))
        elif "autoencode" in args.env:
            env = popgym.envs.autoencode.AutoencodeEasy()
            env = DiscreteAction(Flatten(PreviousAction(env)))
        elif "repeat_previous" in args.env:
            env = popgym.envs.repeat_previous.RepeatPreviousEasy()
            env = DiscreteAction(Flatten(PreviousAction(env)))
        else:
            env = DiscreteAction(Flatten(PreviousAction(gym.make(args.env, render_mode=args.render_mode))))

    elif "FetchReach" in args.env: # E.g. FetchReachDense-v4
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)

        if args.no_reset:
            env = gym.make(args.env, render_mode=args.render_mode, max_episode_steps=args.maxiter)
            if not args.fully_obs:
                env = PartialObsGoal(env, visible_goal_steps=args.visible_goal_steps, reset_goal=args.max_episode_steps)
        else:
            env = gym.make(args.env, render_mode=args.render_mode, max_episode_steps=args.max_episode_steps)
            if not args.fully_obs:
                env = PartialObsGoal(env, visible_goal_steps=args.visible_goal_steps)
        env = gym.wrappers.FlattenObservation(env)

    elif "Mikasa-" in args.env: # E.g. Mikasa-ChainOfColors3-v0
        import mikasa_robo_suite
        # from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info

        env_name = args.env.split("Mikasa-")[1]
        obs_mode = "rgb" if not args.fully_obs else "state"
        env = gym.make(env_name, obs_mode=obs_mode, render_mode=args.render_mode)
        # env = gym.make(env_name, num_envs=args.nenvs, obs_mode=obs_mode, render_mode=args.render_mode)
        obs, _ = env.reset()
        # print(obs["agent"].keys(),"_________________________________________________________________________")
        # state_wrappers_list, episode_timeout = env_info(env_name)
        # print(f"Episode timeout: {episode_timeout}")
        # for wrapper_class, wrapper_kwargs in state_wrappers_list:
        #     env = wrapper_class(env, **wrapper_kwargs)

    elif "MemoryGym-" in args.env: # E.g. MemoryGym-SearingSpotlights-v0
        import memory_gym

        env_name = args.env.split("MemoryGym-")[1]
        env = gym.make(env_name, render_mode=args.render_mode)
        obs, _ = env.reset()
        # print(obs,"_________________________________________________________________________")

    elif "MiniGrid" in args.env: # E.g. MiniGrid-MemoryS7-v0 
        import minigrid
        from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper

        if args.maze_length>3:
            env = gym.make(args.env, size=args.maze_length, render_mode=args.render_mode)#, agent_view_size=3)
        else:
            env = gym.make(args.env, render_mode=args.render_mode)#, agent_view_size=3)
        # env = MiniGridMod(env)
        if args.fully_obs:
            env = FullyObsWrapper(env)
            env = RGBImgObsWrapper(env)
        else:
            env = RGBImgPartialObsWrapper(env)

        # env = OneHotPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        # env = gym.wrappers.FlattenObservation(env)

    elif "bsuite" in args.env:
        import bsuite
        from bsuite.utils import gym_wrapper

        env = gym.make(args.env, render_mode=args.render_mode)
        env = bsuite.load_and_record_to_csv('catch/0', results_dir='/path/to/results')
        env = gym_wrapper.GymFromDMEnv(env)

    else:
        env = gym.make(args.env, render_mode=args.render_mode, max_episode_steps=args.max_episode_steps)
        env = gym.wrappers.FlattenObservation(env)   
        
    ### Memory management approach
    if args.algo not in ["RecurrentPPO"]:
        if args.stack_type=="framestack":
            env = FrameStack(env, args.num_stack)
        elif args.stack_type=="demir":
            env = DemirFrameStack(env, args.num_stack)
        elif "adaptive" in args.stack_type:
            env = AdaptiveStack(env, args.num_stack, multi_head=not args.single_head)
    else:
        if args.stack_type=="framestack":
            env = FrameStack(env, 1)
        elif "adaptive" in args.stack_type:
            env = AdaptiveStack(env, args.num_stack, multi_head=not args.single_head, observe_stack=False)
    
    if args.arch == 'cnn' or args.arch == 'mlp': 
        env = RGBObsStacktoChannel(env)
    

    return env, name
