"""Wrapper that stacks frames, and additional helper wrappers."""
from collections import deque, defaultdict
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict 


class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    Standard environment wrapper for Frame Stacking (Sliding Window)
    
    :param env: Gymnasium environment
    :type env: gym.Env
    :param num_stack: Memory stack length
    :type num_stack: int
    :param observe_stack: If False, only modify the action space and let the RL algorithm 
        implementation handle the stack management based on these actions. 
        Usefull for memory efficient implementations (to avoid duplicate rollout buffers).
    """
    def __init__(
        self,
        env: gym.Env,
        num_stack: int=1,
        observe_stack: bool = True,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, observe_stack=observe_stack,
        )
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.unwrapped.num_stack = num_stack
        self.observe_stack = observe_stack
        self.frames = deque(maxlen=num_stack)
        self.frames_render = deque(maxlen=num_stack)

        if self.observe_stack:
            low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(
                self.observation_space.high[np.newaxis, ...], num_stack, axis=0
            )
            self.observation_space = Box(
                low=low, high=high, dtype=self.observation_space.dtype
            )

    def observation(self, observation):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.array(self.frames)

    def step(self, action):
        if len(self.frames_render)==self.num_stack:
            del self.frames_render[0]

        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.observe_stack:
            self.frames.append(observation)
            observation = self.observation(None)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.frames_render = deque(maxlen=self.num_stack)
        if self.observe_stack:
            [self.frames.append(obs*0) for _ in range(self.num_stack)]
            self.frames.append(obs)
            obs = self.observation(None)

        return obs, info

    def render(self, *args, **kwargs):
        if self.env.render_mode=="rgb_array":
            if len(self.frames_render)==0:
                image = self.env.render(*args, **kwargs)
                self.frames_render = [image * 0 for _ in range(self.num_stack - 1)] 
                self.frames_render.append(image)
            if len(self.frames_render)==self.num_stack-1:
                image = self.env.render(*args, **kwargs)
                self.frames_render.append(image)
            return np.concatenate(self.frames_render, axis=1)
        else:
            self.env.render(*args, **kwargs)


class AdaptiveStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    Environment wrapper for Adaptive Stacking

    :param env: Gymnasium environment
    :type env: gym.Env
    :param num_stack: Memory stack length
    :type num_stack: int
    :param multi_head: For Discrete actions, False->Crossproduct and True->MultiDiscrete. 
        For Continuous (Box) actions, False->Crossproduct and True->Binary
    :param observe_stack: If False, only modify the action space and let the RL algorithm 
        implementation handle the stack management based on these actions. 
        Usefull for memory efficient implementations (to avoid duplicate rollout buffers).
    """
    def __init__(
        self,
        env: gym.Env,
        num_stack: int=1,
        multi_head: bool = True,
        observe_stack: bool = True,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            num_stack=num_stack,
            multi_head=multi_head,
            observe_stack=observe_stack,
        )
        gym.ObservationWrapper.__init__(self, env)

        assert num_stack > 0, "num_stack should be greater than 0"
        self.num_stack = num_stack
        self.unwrapped.num_stack = num_stack
        self.multi_head = multi_head
        self.observe_stack = observe_stack
        self.frames = deque(maxlen=num_stack)
        self.frames_render = deque(maxlen=num_stack)

        if self.observe_stack:
            low = np.repeat(
                self.observation_space.low[np.newaxis, ...], num_stack, axis=0
            )
            high = np.repeat(
                self.observation_space.high[np.newaxis, ...], num_stack, axis=0
            )
            self.observation_space = Box(
                low=low, high=high, dtype=self.observation_space.dtype
            )

        self.unwrapped.memory_dim = None
        self.env_action_space = env.action_space
        if self.multi_head and isinstance(self.env_action_space, Discrete):
            # Separate heads for env action and memory action index
            self.unwrapped.memory_dim = 1
            self.action_space = MultiDiscrete([self.env_action_space.n, num_stack])
        elif isinstance(self.env_action_space, MultiDiscrete):
            # Separate heads for env action and memory action index
            self.unwrapped.memory_dim = len(self.env_action_space.nvec)
            self.action_space = MultiDiscrete(np.append(self.env_action_space.nvec, num_stack))
        elif isinstance(self.env_action_space, Discrete):
            # Single discrete space: env.n * num_stack
            self.action_space = Discrete(self.env_action_space.n * num_stack)
        elif self.multi_head:
            # Continuous case: represent memory action as a binary number (by thresholding reals)
            self.unwrapped.memory_dim = len(self.env_action_space.low)
            low = np.concatenate([
                self.env_action_space.low,
                np.zeros(int(np.ceil(np.log(num_stack))), dtype=np.uint8)+self.env_action_space.low[0],
            ])
            high = np.concatenate([
                self.env_action_space.high,
                np.zeros(int(np.ceil(np.log(num_stack))), dtype=np.uint8)+self.env_action_space.high[0],
            ])
            self.action_space = Box(low=low, high=high, dtype=self.env_action_space.dtype)
        else:
            # Continuous case: represent memory action as a single real number (which is then quantized)
            self.unwrapped.memory_dim = len(self.env_action_space.low)
            low = np.concatenate([
                self.env_action_space.low,
                np.zeros(1, dtype=np.uint8)+self.env_action_space.low[0],
            ])
            high = np.concatenate([
                self.env_action_space.high,
                np.zeros(1, dtype=np.uint8)+self.env_action_space.high[0],
            ])
            self.action_space = Box(low=low, high=high, dtype=self.env_action_space.dtype)

    def split_env_action_memory_action(self, action):
        # Unpack from [env_action, memory_action_index]
        if self.multi_head and isinstance(self.env_action_space, Discrete):
            env_action = int(action[0])
            memory_action = int(action[1])
            return env_action, memory_action
        elif isinstance(self.action_space, MultiDiscrete):
            env_action = action[:-1]
            memory_action = int(action[-1])
            return env_action, memory_action
        elif isinstance(self.action_space, Discrete):
            # Unflatten
            orig_n = self.action_space.n // self.num_stack
            env_action = int(action) % orig_n
            memory_action = int(action) // orig_n
            return env_action, memory_action
        elif self.multi_head:
            action = np.asarray(action)
            env_action = action[:-int(np.ceil(np.log(self.num_stack)))]
            memory_action = action[-int(np.ceil(np.log(self.num_stack))):]
            memory_action = int("".join(["1" if b else "0" for b in memory_action>0]), 2)
        else:
            action = np.asarray(action)
            env_action = action[:-1]
            memory_action = action[-1:]
            memory_action_space = np.linspace(self.env_action_space.low[0],self.env_action_space.high[0],self.num_stack)
            memory_action = np.abs(memory_action_space-memory_action).argmin()
        return env_action, memory_action

    def observation(self, observation):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.array(self.frames)

    def step(self, action):
        info = dict(pre_frames=self.frames.copy())
        action, memory_action = self.split_env_action_memory_action(action)
        if len(self.frames_render)==self.num_stack: # Only used for env.render()
            self.frames_render.pop(memory_action)

        observation, reward, terminated, truncated, env_info = self.env.step(action)
        info.update(env_info)
        info["obs"] = observation.copy()
        if self.observe_stack:
            self.frames.pop(memory_action)
            self.frames.append(observation)
            state = self.observation(observation)
            observation = state
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames_render = [] # Only used for env.render()
        if self.observe_stack:
            self.init_obs = obs * 0
            self.frames = [self.init_obs * 0 for _ in range(self.num_stack - 1)]
            self.frames.append(obs)
            state = self.observation(obs)
            obs = state
        return obs, info

    def render(self, *args, **kwargs):
        if self.env.render_mode=="rgb_array":
            if len(self.frames_render)==0:
                image = self.env.render(*args, **kwargs)
                self.frames_render = [image * 0 for _ in range(self.num_stack - 1)] 
                self.frames_render.append(image)
            if len(self.frames_render)==self.num_stack-1:
                image = self.env.render(*args, **kwargs)
                self.frames_render.append(image)
            return np.concatenate(self.frames_render, axis=1)
        else:
            self.env.render(*args, **kwargs)

class DemirFrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        multi_head: bool = True,
        intrinsic_rewards: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            num_stack=num_stack,
            multi_head=multi_head,
            intrinsic_rewards=intrinsic_rewards,
        )
        gym.ObservationWrapper.__init__(self, env)

        assert num_stack > 0, "num_stack should be greater than 0"
        self.num_stack = num_stack
        self.unwrapped.num_stack = num_stack
        self.multi_head = multi_head
        self.frames = deque(maxlen=num_stack)
        self.frames_render = deque(maxlen=num_stack)
        self.intrinsic_rewards = intrinsic_rewards
        self.beta = 1

        low = np.repeat(
            self.observation_space.low[np.newaxis, ...], num_stack, axis=0
        )
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

        self.env_action_space = env.action_space
        if self.multi_head and isinstance(self.env_action_space, Discrete):
            # Separate heads for env action and memory_action index
            self.action_space = MultiDiscrete([self.env_action_space.n, 2])
        elif isinstance(self.env_action_space, MultiDiscrete):
            # Separate heads for env action and memory_action index
            self.action_space = MultiDiscrete(np.append(self.env_action_space.nvec, 2))
        elif isinstance(self.env_action_space, Discrete):
            # Single discrete space: env.n * num_stack
            self.action_space = Discrete(self.env_action_space.n * 2)
        elif isinstance(self.env_action_space, Box):
            # Continuous case: append scalar memory_action
            low = np.concatenate([
                self.env_action_space.low,
                np.zeros(1, dtype=np.uint8)+self.env_action_space.low[0],
            ])
            high = np.concatenate([
                self.env_action_space.high,
                np.zeros(1, dtype=np.uint8)+self.env_action_space.high[0],
            ])
            self.action_space = Box(low=low, high=high, dtype=self.env_action_space.dtype)
        else:
            raise ValueError(f"Env action space not supported: {self.env_action_space}")

    def split_env_action_memory_action(self, action):
        # Unpack from [env_action, memory_action_index]
        if self.multi_head and isinstance(self.env_action_space, Discrete):
            env_action = int(action[0])
            memory_action = int(action[1])
            return env_action, memory_action
        elif isinstance(self.action_space, MultiDiscrete):
            env_action = action[:-1]
            memory_action = int(action[-1])
            return env_action, memory_action
        elif isinstance(self.action_space, Discrete):
            # Unflatten
            orig_n = self.action_space.n // 2
            env_action = int(action) % orig_n
            memory_action = int(action) // orig_n
            return env_action, memory_action
        else:
            # Continuous: last entries are one-hot memory_action
            action = np.asarray(action)
            env_action = action[:-1]
            memory_action = action[-1:]
            memory_action_space = np.linspace(self.env_action_space.low[0],self.env_action_space.high[0],2)
            # print(memory_action_space, memory_action, np.abs(memory_action_space-memory_action), np.abs(memory_action_space-memory_action).argmin())
            memory_action = np.abs(memory_action_space-memory_action).argmin()
            return env_action, memory_action

    def observation(self, observation):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.array(self.frames)

    def step(self, action):
        info = dict(pre_frames=self.frames.copy())
        action, memory_action = self.split_env_action_memory_action(action)
        if memory_action == 0:
            self.memory_used += 1
            if self.memory_used >= self.num_stack:
                self.memory_used = self.num_stack
                self.frames.pop(0)
                if len(self.frames_render)==len(self.frames)+1:
                    self.frames_render.pop(0)
        else:
            del self.frames[-1]
            if len(self.frames_render)==len(self.frames)+1:
                del self.frames_render[-1]

        observation, reward, terminated, truncated, env_info = self.env.step(action)
        info.update(env_info)
        info["obs"] = observation.copy()
        self.frames.append(observation)
        self.n[tuple(observation)] += 1
        
        if self.intrinsic_rewards:
            if self.memory_used >= self.num_stack: self.h[tuple(observation)] += 1
            # reward = reward + self.beta*(sum([(1-self.n[tuple(obs)]/sum(list(self.n.values())))**self.h[tuple(obs)] for obs in self.frames])/(self.memory_used+1) - 1)
            reward = reward + self.beta*(sum([(1-self.n[tuple(obs)]/sum(list(self.n.values()))) for obs in self.frames]) - self.num_stack-1)

        state = self.observation(observation)
        if self.mod_obs:
            observation = state
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):        
        obs, info = self.env.reset(**kwargs)
        self.n = defaultdict(lambda: 0)
        self.h = defaultdict(lambda: 0)

        self.n[tuple(obs)] += 1
        self.h[tuple(obs)] += 1

        self.memory_used = 0
        self.init_obs = obs * 0
        [self.frames.append(self.init_obs) for _ in range(self.num_stack)]
        self.frames.append(obs)
        self.frames_render = []

        state = self.observation(obs)
        if self.mod_obs:
            obs = state
        return obs, info

    def render(self, *args, **kwargs):
        if self.env.render_mode=="rgb_array":
            if len(self.frames_render)==0:
                image = self.env.render(*args, **kwargs)
                self.frames_render = [image * 0 for _ in range(self.num_stack - 1)] 
                self.frames_render.append(image)
            if len(self.frames_render)==len(self.frames)-1:
                image = self.env.render(*args, **kwargs)
                self.frames_render.append(image)
            return np.concatenate(self.frames_render, axis=1)
        else:
            self.env.render(*args, **kwargs)

class FlatObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=self.observation_space.low.min(), 
                                     high=self.observation_space.high.max(), 
                                     shape=self.observation(self.observation_space.sample()).shape,
                                     dtype=self.observation_space.dtype
        )
        
    def observation(self, observation):
        return observation.flatten()
    

class RGBObsStacktoChannel(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=self.observation_space.low.min(), 
                                     high=self.observation_space.high.max(), 
                                     shape=self.observation(self.observation_space.sample()).shape,
                                     dtype=self.observation_space.dtype
        )
        
    def observation(self, observation):
        if len(observation.shape) == 4: return np.concatenate(observation, axis=1)
        return observation

class TupleObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        
    def observation(self, observation): 
        return tuple(observation.flatten())


class PartialObsGoal(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        visible_goal_steps = 2,
        fetchhideblock = True,
        reset_goal = float("inf"),
        noise = 0,
        new_alpha = 0.2
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.steps = 0
        self.visible_goal_steps = visible_goal_steps
        self.fetchhideblock = fetchhideblock
        self.reset_goal = reset_goal
        self.noise = noise
        self.new_alpha = new_alpha

        # Access MuJoCo internals
        self.model = env.unwrapped.model
        self.goal_site_id = 0

    def step(self, action):
        self.steps += 1
        if self.steps % self.reset_goal == 0:
            self.steps = 0
            self.env.unwrapped.goal = self.env.unwrapped._sample_goal()

        if self.noise:
            action = (1-self.noise)*action + self.noise*self.env.action_space.sample()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation["observation"][5:] *= 0
        if self.steps % self.reset_goal >= self.visible_goal_steps:
            observation["desired_goal"] *= 0
            # if self.steps >= self.visible_goal_steps*2:
            #     observation["observation"][3:6] *= 0
            self.model.site_rgba[self.goal_site_id][3] = self.new_alpha

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.steps = 0
        observation, info = self.env.reset(*args, **kwargs)
        observation["observation"][5:] *= 0

        self.model.site_rgba[self.goal_site_id][3] = 1.0

        return observation, info

class MiniGridMod(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        env.action_space = Discrete(3)
        gym.ObservationWrapper.__init__(self, env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # if terminated:
        #     reward = 1.0 if reward>0 else -1
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)