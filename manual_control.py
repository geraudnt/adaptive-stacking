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

KEYS = {27:'escape', 8:'backspace', 9:'tab', 13:'enter', 32:' ', 81:'left', 82:'up', 83:'right', 84:'down', 117:"U", 100:"D", 108:"L", 114:"R", 102:"F", 98:"B", 120:"X"}
controls = {"Quit":'escape', 'reset':'backspace', 'random action':'tab'}
print(f"\n General controls: {controls} \n")
print(f"\n Memory controls: Press keys 0-{min(9,args.num_stack-1)} to pop the ith observation from the stack (and take a random environment action) \n")

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
    obs, _ = env.reset(seed=args.seed)
    print("Reset")
    return obs
    
def step(action):
    obs, reward, done, truncated, _ = env.step(action)
    print("reward",reward, "done", done, "truncated", truncated)
    return obs

def key_handler(key):
    if key in KEYS:
        key = KEYS[key]
        print('pressed', key)
    
    minigrid_actions = {"left":0, "right":1, "up":2, "enter":3, "down":4, " ":5}
    cube_camera_actions = {"left":"XL", "right":"XR", "up":"XU", "down":"XD"}
    if "cube-v0" == args.env and key in cube_camera_actions:
        key = cube_camera_actions[key]

    try:
        if key == 'backspace': 
            return reset()
        elif hasattr(env.unwrapped,"actions") and key in env.unwrapped.actions: 
            if type(env.unwrapped.actions)==dict:
                action = env.unwrapped.actions[key]
            else: # list
                action = env.unwrapped.actions.index(key)
            if args.stack_type=="adaptive" and not args.single_head:
                action = [action, args.num_stack-1]
                print(f"Memory action i={args.num_stack-1}. Environment action: {action}")
            else:
                print(f"Environment action: {action}")
        elif "MiniGrid" in args.env and key in minigrid_actions: 
            action = minigrid_actions[key]
            if args.stack_type=="adaptive" and not args.single_head:
                action = [action, args.num_stack-1]
                print(f"Memory action i={args.num_stack-1}. Environment action: {action}")
            else:
                print(f"Environment action: {action}")
        else: 
            action = env.action_space.sample()
            if args.stack_type=="adaptive" and key in range(48,58):
                if key-48 < args.num_stack:
                    action[-1] = key-48
                    print(f"Memory action i={key-48}. Random environment action: {action}")
                else:
                    print(f"Unknown pressed key={key}. Random action taken: {action}")
            else:
                print(f"Unknown pressed key={key}. Random action taken: {action}")

        return step(action)
    except:
        action = env.action_space.sample()
        print(f"Unknown pressed key={key}. Random action taken: {action}")
        return step(action)


if __name__ == "__main__": 
    env, _ = make_env(args)
    env.action_space.seed(args.seed)
    print("Observation space: ", env.observation_space)
    print("action_space: ", env.action_space)
    if hasattr(env.unwrapped,"actions"):
        print("actions: ", env.unwrapped.actions)
    # assert ((len(env.observation_space.shape) == 3) and args.agent_view) or (not args.agent_view), "agent_view is only for RGB image observations"  
    
    cv2.namedWindow(args.env, cv2.WINDOW_NORMAL); start()
