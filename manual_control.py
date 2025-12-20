#!/usr/bin/env python3

import sys, cv2, numpy as np
sys.path.insert(1, '../')

import gymnasium as gym
from envs.make_env import make_env

from get_args import parser
args = parser.parse_args()

# For Mujoco envs, Try he following exports if you get GLFW errors: 
# export LD_PRELOAD=""
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

KEYS = {27:'escape', 8:'backspace', 9:'tab', 13:'enter', 32:' ', 81:'left', 82:'up', 83:'right', 84:'down'}
controls = {"Quit":'escape', 'reset':'backspace', 'random action':'tab', 'minigrid_pickup':'enter', 'minigrid_open':' ', 'left':'left', 'up':'up', 'right':'right', 'down':'down'}
print(f"\n Controls: {controls} \n")

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def redraw(img):
    if not args.agent_view: 
        img = env.render()
    if args.agent_view or args.render_mode=="rgb_array":
        # img = cv2.resize(img, dsize=(1024,int(1024*(img.shape[0]/img.shape[1]))), interpolation=cv2.INTER_AREA)
        img = transform_rgb_bgr(img)
        cv2.imshow(args.env, img)   

def start():
    obs = reset()
    while True:
        redraw(obs)
        key = cv2.waitKey(2)
        if key>=0:
            obs = key_handler(key)
        if key == 27: 
            cv2.destroyAllWindows()
            break

def reset():      
    obs, _ = env.reset(seed=args.run)
    print("Reset")
    return obs
    
def step(action):
    obs, reward, done, truncated, _ = env.step(action)
    print("reward",reward, "done", done, "truncated", truncated)
    return obs

def key_handler(key):
    if key not in KEYS: print('Unknown pressed', key); return
    key = KEYS[key]
    print('pressed', key)
    
    # minigrid_actions = {"left":"left", "right": "right", "up":"forward", "enter":"pickup", "down":"drop", " ":"toggle"}
    minigrid_actions = {"left":0, "right":1, "up":2, "enter":3, "down":4, " ":5}

    try:
        if key == 'backspace': 
            return reset()
        elif hasattr(env.unwrapped,"actions") and type(env.unwrapped.actions)==dict and key in env.unwrapped.actions: 
            action = env.unwrapped.actions[key]
        elif "MiniGrid" in args.env and key in minigrid_actions: 
            action = minigrid_actions[key]
        else: 
            action = env.action_space.sample()
            print("Unknown action. Random action taken:", action)
        return step(action)
    except:
        action = env.action_space.sample()
        print("Unknown action. Random action taken:", action)
        return step(action)


if __name__ == "__main__":    
    args.single_head = True
    env, _ = make_env(args)
    print("Observation space: ", env.observation_space)
    print("action_space: ", env.action_space)
    # assert ((len(env.observation_space.shape) == 3) and args.agent_view) or (not args.agent_view), "agent_view is only for RGB image observations"  
    
    cv2.namedWindow(args.env, cv2.WINDOW_NORMAL); start()
