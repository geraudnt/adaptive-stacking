import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble":r'\usepackage{pifont,marvosym,scalerel}'
})



class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    # COLOURS = {"empty": [0.8, 0.8, 0.8, 1], "wall": [0.0, 0.0, 0.0, 1], "start": [0.5, 1, 0.5, 1], "junction": [0.5, 0.5, 1, 1], "top-goal":[1, 0.5, 0.5, 1], "bottom-goal":[0.5, 1, 0.5, 1]}
    COLOURS = {"0": [0.8, 0.8, 0.8, 1], "#": [0.0, 0.0, 0.0, 0.0], "S": [0.4, 1, 0.4, 1], "1": [0.4, 0.4, 1, 1], "2":[1, 0.4, 0.4, 1], "3":[0.4, 1, 0.4, 1]}

    BASEMAP="# # # #\n" \
            "# # 2 #\n" \
            "# S 1 #\n" \
            "# # 3 #\n" \
            "# # # #"
    
    MAP =   "# # # #\n" \
            "# # 2 #\n" \
            "# S 1 #\n" \
            "# # 3 #\n" \
            "# # # #"

    def __init__(self, MAP=MAP, length=2, random_length=False, active=False, fix_start=True, seed=None, max_episode_step=1000000, fully_obs=False, goal_obs=False, continual=False, render_mode = 'human'):   
        self.MAP, self.length, self.active, self.fix_start, self.goal_obs, self.max_episode_step, self.render_mode = MAP, length, active, fix_start, goal_obs, max_episode_step, render_mode
        self.np_random, self.seed = seeding.np_random(seed)
        self.render_params = dict(agent=True, env_map=True, skill=None, policy=False, title=None, cmap="RdYlBu_r")
        
        if length == None: length = 2
        assert length>=2, "Maze length should be >= 2"

        self.length = length
        self.min_length = 2
        self.random_length = random_length
        self.continual = continual
        self.fully_obs = fully_obs
        self.hallwayStates = None
        self.junction, self.goal1, self.goal2 = None, None, None
        self.possiblePositions, self.walls = [], []
        self.n, self.m, self.grid = None, None, None

        row0,row1,row2,row3,row4 = self.BASEMAP.split("\n")
        row0 = row0[:3]+" #"*(self.length-2)+row0[3:]
        row1 = row1[:3]+" #"*(self.length-2)+row1[3:]
        row2 = row2[:3]+" 0"*(self.length-2)+row2[3:]
        row3 = row3[:3]+" #"*(self.length-2)+row3[3:]
        row4 = row4[:3]+" #"*(self.length-2)+row4[3:]
        self.MAP = "\n".join([row0,row1,row2,row3,row4])
        self._map_init()
        self.map_img = self._gridmap_to_img() 

        if active: self.start_position = (2, 2)
        else:      self.start_position = (2, 1)
        # if not self.fix_start: self.start_position = None
        self.goals = [self.goal1,self.goal2]
        self.start_goal = None
        if self.goal_obs: self.start_states = [((0,0),*self.start_position)]
        else:             self.start_states = [(*goal,*self.start_position) for goal in self.goals]

        # Gym spaces for observation and action space
        self.obs_onehot = np.eye(4, dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8)
        self.actions = dict(up = 0, right = 1, down = 2, left = 3)
        self.action_space = spaces.Discrete(len(self.actions),seed=seed)

        # Transition probs
        self.states = []
        for goal in [(0,0),self.goal1,self.goal2]:
            if goal==(0,0) and self.goal_obs: continue
            for position in self.possiblePositions:
                self.states.append((*goal,*position))
        self.num_states = len(self.possiblePositions)*(self.goal_obs+1) if self.fully_obs else 4

    def get_obs(self, state):
        if self.fully_obs: return state
        position = state[2:4]
        state = int(self.grid[position[0]][position[1]])
        state = self.obs_onehot[state].copy()
        return state

    def get_pos_obs(self, state):
        if self.fully_obs: return state
        position = state
        state = int(self.grid[position[0]][position[1]])
        state = self.obs_onehot[state].copy()
        return state

    def stepP(self, state, action, fully_obs=True):
        assert self.action_space.contains(action)
        
        if state[2:4] in [self.goal1, self.goal2]: return [(1,state[2:4],0,True)]

        transitions = []
        goals = [state[:2]] if np.sum(state[:2])>0 else [self.goal1,self.goal2] 
        for goal in goals:
            goal_observed = np.sum(state[:2])>0
            position = state[2:4]
            
            x, y = position
            if not self.active:
                if self.grid[x][y] == "1":
                    action = self.actions['up'] if action in [self.actions['up'],self.actions['right']] else self.actions['down']
                else:
                    action = self.actions['right']
            if action == self.actions['up']: x = x - 1
            elif action == self.actions['right']: y = y + 1
            elif action == self.actions['down']: x = x + 1
            elif action == self.actions['left']: y = y - 1
            next_position = (x, y)
            if self.grid[x][y] != "#": position = next_position # if not wall
            
            reward, done = 0, False
            if position == goal:                       reward, done =  1, True
            elif position in [self.goal1, self.goal2]: reward, done = -1, True
            # else:
            #     # a penalty (when t > o) if x < t - o (desired: x = t - o)
            #     reward = float(y - 1 < self.steps - self.active) * (-1/(self.length))
            
            if self.continual and (position in [self.goal1, self.goal2]): 
                done, goal_observed = False, self.goal_obs
                position = self.start_position
                idx = self.start_goal if self.start_goal!=None else np.random.randint(len(self.goals))
                goal = self.goals[idx]
            state_ = int(self.grid[position[0]][position[1]])
            state_ = self.obs_onehot[state_].copy()
            if position == (2,1): goal_observed = True
            if fully_obs: state_[:2] = np.array(goal)*goal_observed; state_[2:4] = position; state_[4:] *= 0

            transitions.append((1/len(goals),tuple(state_),reward,done)) 

        return transitions

    def step(self, action):
        assert self.action_space.contains(action)    
        
        x, y = self.position
        if not self.active:
            if self.grid[x][y] == "1":
                action = self.actions['up'] if action in [self.actions['up'],self.actions['right']] else self.actions['down']
                # action = self.actions['up'] if self.goal==self.goal1 else self.actions['down']
            else:
                action = self.actions['right']
        if action == self.actions['up']: x = x - 1
        elif action == self.actions['right']: y = y + 1
        elif action == self.actions['down']: x = x + 1
        elif action == self.actions['left']: y = y - 1
        next_position = (x, y)            
        if self.grid[x][y] != "#": self.position = next_position # if not wall

        reward, self.done = 0, False
        if self.position == self.goal:                  reward, self.done =  1, True
        elif self.position in [self.goal1, self.goal2]: reward, self.done = -1, True
        elif False:
            # a penalty (when t > o) if x < t - o (desired: x = t - o)
            reward = float(y - 1 < self.steps - self.active) * (-1/(self.length))

        if self.continual and (self.position in [self.goal1, self.goal2]): 
            self.done, self.goal_observed = False, self.goal_obs
            idx = self.start_goal if self.start_goal!=None else np.random.randint(len(self.goals))
            self.goal = self.goals[idx]
            self.COLOURS["S"] = self.COLOURS[str(idx+2)]
            self.grid[2][1] = self.grid[self.goal[0]][self.goal[1]]
            
            self.position = self.start_position
            if not self.start_position:
                idx = self.np_random.integers(len(self.possiblePositions))
                self.position = self.possiblePositions[idx]

        state = int(self.grid[self.position[0]][self.position[1]])
        state = self.obs_onehot[state].copy()
        # state = np.concatenate([state,[action]])
        if self.position == (2,1): self.goal_observed = True 
        if self.fully_obs: state[:2] = np.array(self.goal)*self.goal_observed; state[2:4] = self.position; state[4:] *= 0

        self.steps += 1
        truncated = False
        # if (not self.continual) and self.max_episode_step and self.steps>=self.max_episode_step: truncated = True  
        if (not self.continual) and self.steps > self.length + self.active: truncated = True  
        
        if self.render_mode=="human":
            self.render()  
                
        return state, reward, self.done, truncated, {"reward":reward}
    
    def reset(self, seed=None, **kwargs):
        self.done, self.goal_observed, self.steps = False, self.goal_obs, 0        
        self.np_random, self.seed = seeding.np_random(seed)
        if self.random_length:
            length = self.np_random.integers(self.min_length,self.length+1)
            junction = self.grid[self.junction[0]][self.junction[1]]
            goal1 = self.grid[self.goal1[0]][self.goal1[1]]
            goal2 = self.grid[self.goal2[0]][self.goal2[1]]
            self.grid[self.junction[0]][self.junction[1]] = "0"
            self.grid[self.goal1[0]][self.goal1[1]] = "#"
            self.grid[self.goal2[0]][self.goal2[1]] = "#"
            self.possiblePositions.remove(self.goal1)
            self.possiblePositions.remove(self.goal2)
            self.junction, self.goal1, self.goal2 = (2,length), (1,length), (3,length)
            self.grid[self.junction[0]][self.junction[1]] = junction
            self.grid[self.goal1[0]][self.goal1[1]] = goal1
            self.grid[self.goal2[0]][self.goal2[1]] = goal2
            self.possiblePositions.append(self.goal1)
            self.possiblePositions.append(self.goal2)
            
            self.goals = [self.goal1,self.goal2]
            self.start_goal = None
            if self.goal_obs: self.start_states = [((0,0),*self.start_position)]
            else:             self.start_states = [(*goal,*self.start_position) for goal in self.goals]
            # Transition probs
            self.states = []
            for goal in [(0,0),self.goal1,self.goal2]:
                if goal==(0,0) and self.goal_obs: continue
                for position in self.possiblePositions:
                    self.states.append((*goal,*position))

        
        idx = self.start_goal if self.start_goal!=None else np.random.randint(len(self.goals))
        self.goal = self.goals[idx]
        self.COLOURS["S"] = self.COLOURS[str(idx+2)]
        self.grid[2][1] = self.grid[self.goal[0]][self.goal[1]]

        self.position = self.start_position
        if not self.start_position:
            idx = self.np_random.integers(len(self.possiblePositions))
            self.position = self.possiblePositions[idx]
        state = int(self.grid[self.position[0]][self.position[1]])
        state = self.obs_onehot[state].copy()
        # state = np.concatenate([state,[0]])
        if self.position == (2,1): self.goal_observed = True 
        if self.fully_obs: state[:2] = np.array(self.goal)*self.goal_observed; state[2:4] = self.position; state[4:] *= 0

        if self.render_mode=="human":
            self.render()
            
        return state, {}

    def render(self):    
        self.fig = self.gen_fig()
        self.fig.canvas.draw()
        if self.render_mode == 'rgb_array':
            height, width = self.fig .get_size_inches() * self.fig.get_dpi()
            img = np.frombuffer(self.fig .canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(int(width), int(height), 3)
            plt.close(self.fig )
            return img
        else:
            plt.pause(0.001) 
    
    def gen_fig(self):
        fig = plt.figure("render", figsize=(self.n, self.m), dpi=100, facecolor='w', edgecolor='k')
        params = {'font.size': 40}
        plt.rcParams.update(params)
        plt.clf(); plt.xticks([]); plt.yticks([]); plt.grid(False)
        if self.render_params["title"]: fig.suptitle(self.render_params["title"], usetex=False)
        plt.tight_layout(pad=0)
        
        s = 0.98
        plt.imshow(self._gridmap_to_img(), origin="upper", extent=[s, self.n-s, self.m-s, s])
        ax = fig.gca(); ax.axis('off')     
        for (y, x) in self.possiblePositions:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, lw=4, ec='black', fc=(0, 0, 0, 0), transform=ax.transData, zorder=10))    
        if self.render_params["agent"]:
            y, x = self.position
            ax.add_patch(patches.Circle((x+0.5, y+0.5), radius=0.4, lw=0.2, ec='white', fc='black', transform=ax.transData, zorder=10))        
        # if self.render_params["skill"]:
        #     v = np.zeros((self.m,self.n))+float("-inf")
        #     for y, x in self.possiblePositions:
        #         env_state, proposition = np.array([y,x], dtype=np.uint8), np.zeros(len(self.predicate_latex), dtype=np.uint8)
        #         tp_state = {'env_state': env_state, 'constraint': proposition}
        #         action, v[y,x] = self.render_params["skill"].get_action(tp_state)  
        #         if self.render_params["policy"]:
        #             self._draw_action(ax, x, y, action%self.action_space.n)
        #     c = ax.imshow(v, origin="upper", cmap=self.render_params["cmap"], extent=[0, self.n, self.m, 0])
            # fig.colorbar(c, ax=ax)
        
        return fig

    def _draw_action(self, ax, x, y, action, color='black'):
        if action == self.actions["up"]:    x += 0.5; y += 1; dx = 0; dy = -0.4
        if action == self.actions["right"]: y += 0.5; dx = 0.4; dy = 0
        if action == self.actions["down"]:  x += 0.5; dx = 0; dy = 0.4
        if action == self.actions["left"]:  x += 1; y += 0.5; dx = -0.4; dy = 0
        ax.add_patch(ax.arrow(x, y, dx, dy, fc=color, ec=color, width=0.005, head_width=0.4))

    def _map_init(self):
        self.grid = []
        lines = self.MAP.split('\n')
        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError("Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(col)
                if col == "#": self.walls.append((i, j))
                else: self.possiblePositions.append((i, j))
                if col == "1": self.junction = (i, j)
                elif col == "2": self.goal1 = (i, j)
                elif col == "3": self.goal2 = (i, j)
            self.grid.append(rowArray)
        self.m = i + 1

    def _gridmap_to_img(self):
        row_size, col_size = len(self.grid), len(self.grid[0])
        img = np.zeros([row_size-2, col_size-2, 4])
        for i in range(row_size-2):
            for j in range(col_size-2):  
                img[i:(i + 1), j:(j + 1)] = self.COLOURS[self.grid[i+1][j+1]]
        return img


gym.envs.registration.register(
    id='tmaze-v0',
    entry_point=GridWorldEnv,
)
