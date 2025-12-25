from algos.grpo import GRPO
from algos.ppo_recurrent import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from collections import defaultdict
from envs.make_env import *
from utils_sb3 import *

from get_args import parser
args = parser.parse_args()


if __name__ == "__main__":
    # Instantiate envs
    env, name = make_env(args)
    log_dir = args.path + name

    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)
    print("log_dir", log_dir)

    env = LoggerWrapper(env, args.stack_type)
    vec_env = env
    if args.n_envs>1:
        def make_env_(rank: int, seed: int = 0):
            def _init():
                if rank==0: 
                    env = LoggerWrapper(make_env(args)[0], args.stack_type)
                    env_ = env
                else:       env_ = make_env(args)[0]
                if args.seed: env_.reset(seed=args.seed + rank)
                else:        env_.reset()
                return env_
            set_random_seed(seed)
            return _init
        vec_env = SubprocVecEnv([make_env_(i) for i in range(args.n_envs)])

    # Instantiate and train model
    policy_type, policy_kwargs = get_policy_type(args)
    if args.algo == "PPO":
        model = PPO(policy_type, vec_env, policy_kwargs=policy_kwargs, device=args.device,
                    n_steps=args.n_steps, batch_size=args.batch_size, seed=args.seed, # n_epochs=20,
                    verbose=1, tensorboard_log=log_dir)
        model.learn(int(args.maxiter), callback=SaveLogCallback(log_dir=log_dir))
    
    elif args.algo == "RecurrentPPO":
        model = RecurrentPPO(policy_type, vec_env, 
                             n_stack = args.num_stack, # Recurrence/Sequence/Context length for the RNN (i.e. sliding window if using Frame Stacking)
                             adaptive_stack=args.stack_type=="adaptive", # Whether to use Adaptive Stacking or the default Frame Stacking (sliding window)
                             policy_kwargs=policy_kwargs, device=args.device,
                             n_steps=args.n_steps, batch_size=args.batch_size, seed=args.seed,
                             verbose=1, tensorboard_log=log_dir)
        model.learn(int(args.maxiter), callback=SaveLogCallback(log_dir=log_dir))
    
    elif args.algo == "GRPO":
        model = GRPO(policy_type, vec_env, policy_kwargs=policy_kwargs, device=args.device,
                    n_steps=args.n_steps, batch_size=args.batch_size, seed=args.seed,
                    verbose=1, tensorboard_log=log_dir)
        model.learn(int(args.maxiter), callback=SaveLogCallback(log_dir=log_dir))
    
    elif args.algo == "QL":
        print_freq, best, gamma, alpha, epsilon = 10000, -float("inf"), 0.99, 0.1, 0.1
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        state, _ = env.reset()
        for steps in range(args.maxiter):     
            state = tuple(state.flatten())   
            if np.random.random() > epsilon: action = np.random.choice(np.flatnonzero(Q[state] == Q[state].max()))
            else:                            action = np.random.randint(env.action_space.n)   
            state_, reward, done, truncate, info = env.step(action)
            best = SaveLogQL(env, Q, best, steps, log_dir, print_freq)
            
            G = 0 if done else np.max(Q[tuple(state_.flatten())])
            Q[state][action] += alpha*(reward + gamma*G - Q[state][action])

            state = state_
            if done or truncate:
                state, _ = env.reset()
            
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")
