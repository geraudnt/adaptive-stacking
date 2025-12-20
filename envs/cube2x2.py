import csv
import gymnasium as gym
from gymnasium import error, utils
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
import random
import copy
import envs.py222 as py222
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# COLOR_MAP = {
#     0: np.array([255, 111, 111]) / 255,  # red (U)
#     1: np.array([255, 255, 255]) / 255,  # white (R)
#     2: np.array([111, 125, 255]) / 255,  # blue (F)
#     3: np.array([111, 255, 143]) / 255,  # green (D)
#     4: np.array([207, 207, 207]) / 255,  # gray (L)
#     5: np.array([26, 26, 26]) / 255,     # black (B)
# }


class cube(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    COLOR_MAP = {
        0: "#ff6f6f",  # top
        1: "#ffffff",  # right
        2: "#6f7dff",  # front
        3: "#6fff8f",  # bottom
        4: "#cfcfcf",  # left
        5: "#1a1a1a",  # back
    }

    FACES_MAP = {
        "U": ((0, 1, 2, 3),   "z",  1),
        "D": ((12,13,14,15),  "z", -1),
        "F": ((8, 9,10,11),   "y",  1),
        "B": ((20,21,22,23),  "y", -1),
        "L": ((16,17,18,19),  "x", -1),
        "R": ((4, 5, 6, 7),   "x",  1),
    }
    
    # Net layout: (row, col) → state index
    grid = {
        (0, 2): 0,  (0, 3): 1,
        (1, 2): 2,  (1, 3): 3,

        (2, 0): 16, (2, 1): 17, (2, 2): 8,  (2, 3): 9,  (2, 4): 4,  (2, 5): 5,  (2, 6): 20, (2, 7): 21,
        (3, 0): 18, (3, 1): 19, (3, 2): 10, (3, 3): 11, (3, 4): 6,  (3, 5): 7,  (3, 6): 22, (3, 7): 23,

        (4, 2): 12, (4, 3): 13,
        (5, 2): 14, (5, 3): 15,
    }

    def __init__(self,episode_steps=1000,scramble_steps=20,random_length=False,cube_cam="full",render_mode = 'human', render_2d=False, render_alpha = 0.5):
        self.episode_steps = episode_steps
        self.scramble_steps = scramble_steps
        self.random_length = random_length
        self.render_alpha = render_alpha
        self.cube_cam = cube_cam
        self.render_mode = render_mode
        self.render_mode = render_mode
        self.render_2d = render_2d
        self.visible_line_width = 2
        self.invisible_line_width = 0
        self.faces = ["U","R","F","D","L","B"]
        self.actions = ["U","U'","R","R'","F","F'","D","D'","L","L'","B","B'"]
        
        self.steps = 0
        self.camera_view = 0
        self.goal_state = py222.initState()
        
        # Hard-coded to be cleaned
        self.camera_views = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
        self.observation_space = spaces.Box(low=0, high=5, shape=(len(self.goal_state),), dtype=np.uint8)
        self.camera_action_views = lambda action,camera_view: camera_view
        if cube_cam == "face":
            self.actions = ["U","U'","R","R'","F","F'","D","D'","L","L'","B","B'","CU","CR","CD","CL"]
            self.camera_action_views = lambda action,camera_view:  {12: [5,0,0,2,0,0], # CU
                                                                    13: [1,2,1,1,2,4], # CR
                                                                    14: [2,3,3,0,3,3], # CD
                                                                    15: [4,5,4,4,5,1], # CL
                                                                    }[action][camera_view]
            self.camera_views = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
            self.observation_space = spaces.Box(low=0, high=5, shape=(4,), dtype=np.uint8)
        elif cube_cam == "orthographic":
            self.actions = ["U","U'","R","R'","F","F'","D","D'","L","L'","B","B'","C"]
            self.camera_action_views = lambda action,camera_view: (camera_view+1)%2
            self.camera_views = [[0,1,2,3,4,5,6,7,8,9,10,11],[12,13,14,15,16,17,18,19,20,21,22,23]]
            self.observation_space = spaces.Box(low=0, high=5, shape=(4*3,), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))

    def step(self, action_ind):
        self.steps += 1
        if action_ind>=12:
            if type(action_ind) != int: action_ind = int(action_ind)
            self.camera_view = self.camera_action_views(action_ind,self.camera_view)
        else:
            action_str = self.actions[action_ind]
            self.state = py222.doAlgStr(self.state, action_str)
        diff = abs(self.goal_state-self.state).sum()

        if diff == 0:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        truncate = False
        if self.steps >= self.episode_steps:
            truncate = True
            
        obs = self.state[self.camera_views[self.camera_view]]
        
        if self.render_mode=="human":
            self.render() 

        return obs, reward, done, truncate, {}

    def reset(self,seed=None,options=None):   
        self.np_random, self.seed = seeding.np_random(seed)

        scramble_steps = self.scramble_steps
        if self.random_length:
            scramble_steps = self.np_random.integers(1,self.scramble_steps)

        self.steps = 0
        state = py222.initState()
        for i in range(scramble_steps):
            action_ind = self.np_random.integers(0,len(self.actions)-1)
            action_str = self.actions[action_ind]
            state = py222.doAlgStr(state, action_str)

        self.state = state
        self.camera_view = 0
        obs = self.state[self.camera_views[self.camera_view]]
        
        if self.render_mode=="human":
            self.render() 
            
        return obs, {}

    def render(self): 
        if self.render_2d:
            self.render_cube_2d()
        else:
            self.render_cube_3d()

        if self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            height, width = self.fig .get_size_inches() * self.fig.get_dpi()
            img = np.frombuffer(self.fig .canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(int(width), int(height), 3)
            # plt.close(self.fig )
            return img
        else:
            self.fig.canvas.draw_idle()
            plt.pause(0.001) 
    
    def render_cube_2d(self):
        if not hasattr(self, "fig"):
            self.polygons = {}
            self.fig = plt.figure("render", figsize=(5, 4))
            self.ax = self.fig.add_axes([0, 0, 1, 1])  # fill entire figure
            self.ax.axis('off')  
            self.ax.set_aspect("equal")    
            self.ax.set_xlim(0, 8)
            self.ax.set_ylim(-5, 1)  
            plt.tight_layout(pad=0)

        square_size = 1
        for (r, c), idx in self.grid.items():
            alpha = 1
            if idx not in self.camera_views[self.camera_view]:
                alpha = self.render_alpha
            if ((r, c), idx) not in self.polygons:
                rect = patches.Rectangle(
                    (c * square_size, -r * square_size),
                    square_size,
                    square_size,
                )
                self.ax.add_patch(rect)    
                self.polygons[((r, c), idx)] = rect
            self.polygons[((r, c), idx)].set_facecolor(self.COLOR_MAP[self.state[idx]])
            self.polygons[((r, c), idx)].set_edgecolor("black")
            self.polygons[((r, c), idx)].set_linewidth(self.visible_line_width if alpha==1 else self.invisible_line_width)
            self.polygons[((r, c), idx)].set_alpha(alpha)  
    
    def render_cube_3d(self):
        new_fig = False
        if not hasattr(self, "fig"):
            new_fig = True
            self.polygons = {}
            if self.render_mode == 'human':
                plt.ion()
            self.fig = plt.figure("render", figsize=(4, 4))
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.set_box_aspect((1, 1, 1))
            self.ax.axis("off") 
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-1.1, 1.1)
            self.ax.set_zlim(-1.1, 1.1)
            self.ax.view_init(elev=25, azim=35)
            plt.tight_layout(pad=0)  

        # Draw visible faces
        for idxs, axis, val in self.FACES_MAP.values():
            colors = [self.COLOR_MAP[self.state[i]] for i in idxs]
            alpha = 1
            if idxs[0] not in self.camera_views[self.camera_view]:
                alpha = self.render_alpha
            self.draw_face(idxs, axis, val, colors, alpha=alpha)  

        # Camera setup        
        if self.render_mode == 'human' and new_fig:
            plt.show(block=False)
        elif self.render_mode != 'human':
            if self.cube_cam == "orthographic":
                if self.camera_view == 0:
                    self.ax.view_init(elev=25, azim=35)
                else:
                    self.ax.view_init(elev=-155, azim=35)
            elif self.cube_cam == "face":
                face_views = {
                    "F": (0, 90),
                    "B": (-180, 90),
                    "U": (90, 0),
                    "D": (-90, 0),
                    "R": (0, 0),
                    "L": (0, -180),
                }
                face = self.faces[self.camera_view]
                elev, azim = face_views[face]
                self.ax.view_init(elev=elev, azim=azim)
                self.ax.set_xlim(-0.8, 0.8)
                self.ax.set_ylim(-0.8, 0.8)
                self.ax.set_zlim(-0.8, 0.8)


    def draw_face(self, idxs, axis, value, colors, alpha=1):
        coords = [-1, 0, 1]
        k = 0
        for i in range(2):
            for j in range(2):
                if (idxs,i,j) not in self.polygons:
                    if axis == "x":
                        verts = [
                            (value, coords[j],   coords[i]),
                            (value, coords[j+1], coords[i]),
                            (value, coords[j+1], coords[i+1]),
                            (value, coords[j],   coords[i+1]),
                        ]
                    elif axis == "y":
                        verts = [
                            (coords[j], value,   coords[i]),
                            (coords[j+1], value, coords[i]),
                            (coords[j+1], value, coords[i+1]),
                            (coords[j], value,   coords[i+1]),
                        ]
                    else:  # z
                        verts = [
                            (coords[j],   coords[i],   value),
                            (coords[j+1], coords[i],   value),
                            (coords[j+1], coords[i+1], value),
                            (coords[j],   coords[i+1], value),
                        ]

                    poly = Poly3DCollection([verts])
                    self.polygons[(idxs,i,j)] = poly
                    self.ax.add_collection3d(poly)
                self.polygons[(idxs,i,j)].set_facecolor(colors[k])
                self.polygons[(idxs,i,j)].set_edgecolor("black")
                self.polygons[(idxs,i,j)].set_linewidth(self.visible_line_width if alpha==1 else self.invisible_line_width)
                self.polygons[(idxs,i,j)].set_alpha(alpha)
                k += 1  
                
        
register(
    id='cube-v0',
    entry_point='envs.cube2x2:cube',
)
