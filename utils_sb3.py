from collections import defaultdict
import numpy as np
import torch, torch.nn as nn

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


def load_agent(algorithm, model_path, env, device="cpu", deterministic=True):
    if algorithm == "QL":
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        Q.update(np.load(model_path, allow_pickle=True).item())
        return lambda s: Q[tuple(s.flatten())].argmax()
    
    elif algorithm == "PPO":
        from stable_baselines3 import PPO
        
        model = PPO.load(model_path, env=env, device=device)
        return lambda s: model.predict(s, deterministic=deterministic)[0]
    
    elif algorithm == "RecurrentPPO":
        from algos.ppo_recurrent import RecurrentPPO
        
        model = RecurrentPPO.load(model_path, env=env, device=device)
        def agent(s, lstm_states=None, lstm_episode_starts=None):
            a, state = model.predict(s, state=lstm_states, episode_start=lstm_episode_starts, deterministic=deterministic)
            return a, state
        return agent
    
    elif algorithm == "GRPO":
        from algos.grpo import GRPO
        
        model = GRPO.load(model_path, env=env, device=device)
        return lambda s: model.predict(s, deterministic=deterministic)[0]

    raise ValueError(f"Unknown algorithm {algorithm!r}")

def SaveLogQL(env, Q, best, steps, log_dir, print_freq):
    stats = env.get_stats()
    
    if steps % print_freq == 0:
        if len(stats["rewards"]) > 0:
            returns = np.sum(stats["rewards"][-1000:])
            if returns >= best:
                best = returns
                np.save(log_dir+"_values", dict(Q))
        np.save(log_dir, stats)

        print("--------------------------------------------") 
        print("timestep", steps)
        print("states", len(list(Q.keys())))
        if len(stats["rewards"]) > 0:
            total_rewards = np.sum(stats["rewards"][-1000:])
            print("total rewards (1000 steps)", total_rewards)
        if len(stats["success"]) > 0:
            successes = np.mean(stats["success"][-100:])
            print("successes", round(successes,2))
        if len(stats["memory_regret"]) > 0:
            mean_passive_count = np.mean(stats["passive_count"][-100:])
            mean_active_count = np.mean(stats["active_count"][-100:])
            mean_memory_regret = np.mean(stats["memory_regret"][-100:])
            print("Passive regret", round(mean_passive_count,2))
            print("Active regret", round(mean_active_count,2))
            print("Memory regret", round(mean_memory_regret,2))
        print("--------------------------------------------") 
    return best

class SaveLogCallback(BaseCallback):
    def __init__(self, print_freq=1000, log_dir="", seed=None, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq, self.log_dir, self.seed  = print_freq, log_dir, seed
        self.best = -float("inf")
        
    def _on_step(self) -> bool:
        stats = self.training_env.env_method("get_stats", indices=[0])[0]
        
        if self.n_calls % self.print_freq == 0:
            if len(stats["rewards"]) > 0:
                returns = np.sum(stats["rewards"][-1000:])
                if returns >= self.best:
                    self.best = returns
                    self.model.save(self.log_dir + "_values")
            np.save(self.log_dir, stats)
        
        if len(stats["rewards"]) > 0:
            total_rewards = np.sum(stats["rewards"][-1000:])
            self.logger.record("total rewards (1000 steps)", total_rewards)
        if len(stats["success"]) > 0:
            total_rewards = np.mean(stats["success"][-100:])
            self.logger.record("success rate", total_rewards)
        if len(stats["memory_regret"]) > 0:
            mean_passive_count = np.mean(stats["passive_count"][-100:])
            mean_active_count = np.mean(stats["active_count"][-100:])
            mean_memory_regret = np.mean(stats["memory_regret"][-100:])
            self.logger.record("Passive regret", mean_passive_count)
            self.logger.record("Active regret", mean_active_count)
            self.logger.record("Memory regret", mean_memory_regret)
        return True


class LoggerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, stack_type ):
        self.stats = {"R":[], "T":0, "episode":[], "regrets":[], "returns":[], "rewards":[], "success":[],"steps":[],"states":[],"learned":[], "memory_regret":[], "passive_count":[], "active_count":[]}
        
        super().__init__(env)
        self.stack_type = stack_type
            
        self.k=0
        self.T=0
        self.t=0
        self.passive_count=0
        self.active_count=0
        self.memory_regret=0
        self.rewards=0
        self.successes=0
        self.log_freq = 100
    
    def get_stats(self):
        return self.stats
    
    def step(self, action):
        state, reward, done, truncated, info = super().step(action)
        gamma = 0.99
        self.rewards += (gamma**self.t)*reward
        if "is_success" in info:
            self.successes += info["is_success"]
        elif "success" in info:
            self.successes += info["success"]
        else:
            self.successes += reward>0
        # self.successes += reward>0
        self.stats["rewards"].append(reward)
        if hasattr(self.env.unwrapped,"MAP"): ### TMaze
            env = None
            if hasattr(self.env,"num_stack"): env = self.env
            elif hasattr(self.env.env,"num_stack"): env = self.env.env
            elif hasattr(self.env.env.env,"num_stack"): env = self.env.env.env
            elif hasattr(self.env.env.env.env,"num_stack"): env = self.env.env.env.env
            if env and len(self.state.shape)==2:
                s_obs = tuple(self.state[-1])
                ns_obs = tuple(state[-1:])
                s_mem = [tuple(self.state[i]) for i in range(0,env.num_stack-1)]
                ns_mem = [tuple(state[i]) for i in range(0,env.num_stack-1)]
                goal = tuple(self.env.unwrapped.get_pos_obs(self.env.unwrapped.goal))
                # print(goal, len(state))
                self.memory_regret += (not done) and (goal not in ns_mem)
                if not done and (goal in s_mem): self.passive_count += (goal not in ns_mem)
                if not done and (s_obs==goal) and (goal not in s_mem): self.active_count += (goal not in ns_mem)
            if self.stack_type == "no_stack" and (tuple(self.state)==goal): self.passive_count += 1; self.active_count += 1; 
            if self.stack_type == "no_stack": self.memory_regret += 1
        
            if (self.T)%self.log_freq==0:
                self.stats["returns"].append(self.rewards)
                self.stats["memory_regret"].append(self.memory_regret)
                self.stats["passive_count"].append(self.passive_count)
                self.stats["active_count"].append(self.active_count)
                self.rewards=0
                self.passive_count=0
                self.active_count=0
                self.memory_regret=0
        self.state = state.copy()

        self.t+=1; self.T += 1
        return state, reward, done, truncated, info
    
    def reset(self, *args, **kwargs):
        state, info = self.env.reset(*args,**kwargs)
        self.state = state.copy()
        self.stats["T"] = self.T
        self.stats["steps"].append(self.t)
        self.stats["episode"].append(self.k-1)
        self.stats["success"].append(self.successes)
        self.t=0
        self.k+=1
        self.successes=0
        
        return state, info

# class MinigridFeaturesExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
#         super().__init__(observation_space, features_dim)
#         self.n_stack = observation_space.shape[0]
#         n_input_channels = observation_space.shape[-1]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 16, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         # Compute shape by doing one forward pass
#         with torch.no_grad(): n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None][:,0,:,:,:]).float().permute(0,3,1,2)).shape[1]
#         self.linear = nn.Sequential(nn.Linear(n_flatten*self.n_stack, features_dim), nn.ReLU())

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         features = torch.cat([self.cnn(observations[:,o,:,:,:].permute(0,3,1,2)) for o in range(self.n_stack)], dim=1)
#         return self.linear(features)


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad(): n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# class CNNFeaturesExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
#         super().__init__(observation_space, features_dim)
#         self.num_stack = observation_space.shape[0]
#         c = observation_space.shape[-1]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(c,  32, kernel_size=8, stride=4), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with torch.no_grad(): n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None][:,0,:,:,:]).float().permute(0,3,1,2)).shape[1]
#         self.linear = nn.Sequential(nn.Linear(n_flatten*self.num_stack, features_dim))
#         # self.linear = nn.Sequential(nn.Linear(n_flatten*self.num_stack, features_dim), nn.ReLU(), nn.Linear(features_dim, features_dim), nn.ReLU(), nn.Linear(features_dim, features_dim), nn.ReLU())

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         features = torch.cat([self.cnn(observations[:,o,:,:,:].permute(0,3,1,2)) for o in range(self.num_stack)], dim=1)
#         return self.linear(features)
    

class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad(): n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU(),)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# Decoder-only Transformer-based feature extractor for sequence observations
class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 embed_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 seq_length: int = 1, with_cnn: bool=False) -> None:
        super().__init__(observation_space, features_dim)
        self.seq_length = seq_length
        # Flatten all dims after sequence dimension
        input_shape = observation_space.shape
        self.feature_dim = int(np.prod(input_shape[1:])) if len(input_shape) > 1 else input_shape[0]
        self.num_stack = input_shape[0]
        self.with_cnn = with_cnn
        if self.with_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(input_shape[-1],  32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad(): self.feature_dim = self.cnn(torch.as_tensor(observation_space.sample()[None][:,0,:,:,:]).float().permute(0,3,1,2)).shape[1]
        # Embedding layer
        self.embed = nn.Linear(self.feature_dim, embed_dim)
        # Learnable positional encodings
        self.pos_emb = nn.Parameter(torch.zeros(seq_length, embed_dim))
        # Decoder-only transformer: stack of TransformerDecoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Projection to desired feature dimension
        self.linear = nn.Linear(embed_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.with_cnn:
            observations = torch.cat([self.cnn(observations[:,o,:,:,:].permute(0,3,1,2)) for o in range(self.num_stack)], dim=1)
        # observations shape: [batch, seq_length, ...]
        batch_size = observations.shape[0]
        # Flatten per-timestep dimensions
        x = observations.view(batch_size, self.seq_length, -1)
        # Apply embedding and add positional encoding
        x = self.embed(x) + self.pos_emb.unsqueeze(0)
        # TransformerDecoder expects [seq_length, batch, embed_dim]
        x = x.permute(1, 0, 2)
        # Create causal mask so decoder attends only to past tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_length).to(x.device)
        # Use the same sequence as both target and memory to simulate decoder-only behavior
        x = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        # Back to [batch, seq_length, embed_dim]
        x = x.permute(1, 0, 2)
        # Mean pooling over sequence
        x = x.mean(dim=1)
        return self.linear(x)
    
class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simple LSTM over a sequence of stacked observations.
    Takes the final hidden state and linearly projects to features_dim.
    """
    def __init__(self,
                 observation_space: gym.Space,
                 features_dim: int = 256,
                 hidden_size: int = 128,
                 lstm_layers: int = 1,
                 seq_length: int = 1,
                 with_cnn: bool=False) -> None:
        super().__init__(observation_space, features_dim)
        self.seq_length = seq_length
        # flatten all dims after the time axis
        self.with_cnn = with_cnn
        if self.with_cnn:
            channels = obs_shape[-1]
            self.cnn = nn.Sequential(
                nn.Conv2d(channels,  32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad(): self.input_dim = self.cnn(torch.as_tensor(observation_space.sample()[None][:,0,:,:,:]).float().permute(0,3,1,2)).shape[1]
        else:
            obs_shape = observation_space.shape  # e.g. (seq_length, …)
            self.input_dim = int(np.prod(obs_shape[1:])) if len(obs_shape) > 1 else 1
        
        # a single‐layer LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # project the last hidden state to features_dim
        self.linear = nn.Linear(hidden_size, features_dim)

    def forward(self, observations) -> torch.Tensor:
        if self.with_cnn:
            observations = torch.cat([self.cnn(observations[:,o,:,:,:].permute(0,3,1,2)) for o in range(self.seq_length)], dim=1)
        # observations: [batch, seq_length, *obs_dims]
        x = observations.view(observations.size(0), self.seq_length, -1)  # [B, T, input_dim]
        _, (h_n, _) = self.lstm(x)
        # take the top‐layer final hidden state
        h_last = h_n[-1]  # [batch, hidden_size]
        return self.linear(h_last)
    

# Select policy architecture and kwargs
def get_policy_type(args):
    if args.algo == 'RecurrentPPO':
        if args.arch == 'mlp':
            policy_type = "MlpLstmPolicy"
        elif args.arch == 'cnn':
            policy_type = "CnnLstmPolicy"
        policy_kwargs = dict(
            # lstm_hidden_size=args.features_dim, 
            lstm_hidden_size=args.hidden_size, 
            n_lstm_layers=2,
            # shared_lstm=True,
        )
    else:
        if args.arch == 'cnn':
            policy_type = "CnnPolicy"
            policy_kwargs = dict(net_arch=[args.hidden_size, args.hidden_size, args.hidden_size])
        elif args.arch == 'mlp':
            policy_type = "MlpPolicy"
            policy_kwargs = dict(net_arch=[args.hidden_size, args.hidden_size, args.hidden_size])
            if args.with_cnn:
                policy_kwargs = dict(
                    net_arch=[args.hidden_size, args.hidden_size, args.hidden_size],
                    features_extractor_class=CNNFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=args.features_dim)
                )
        elif args.arch == 'transformer':
            policy_type = "MlpPolicy"
            policy_kwargs = dict(
                features_extractor_class=TransformerFeaturesExtractor,
                features_extractor_kwargs=dict(
                    features_dim=args.features_dim,
                    embed_dim=args.hidden_size,
                    num_layers=args.num_layers,
                    num_heads=4,
                    seq_length=args.num_stack,
                    with_cnn = args.with_cnn
                ),
            )
        elif args.arch == 'lstm':
            # use MlpPolicy but swap in LSTMFeaturesExtractor
            policy_type = "MultiInputPolicy"
            policy_kwargs = dict(
                features_extractor_class=LSTMFeaturesExtractor,
                features_extractor_kwargs=dict(
                    features_dim=args.features_dim,
                    hidden_size=args.hidden_size,
                    lstm_layers=args.num_layers,
                    seq_length=args.num_stack,
                    with_cnn = args.with_cnn
                ),
            )
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")
    
    return policy_type, policy_kwargs
