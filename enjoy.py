import os, cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt

from envs.make_env import make_env
from utils_sb3 import load_agent

from get_args import parser
args = parser.parse_args()


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

if __name__ == "__main__":    
    # Instantiate env and agent
    env, name = make_env(args)
    if args.load_path:
        agent = load_agent(args.algo, args.load_path, env, args.device, deterministic=False)
    else:
        load_path = args.path + name + ("_values.npy" if args.algo=="QL" else "_values")
        if not os.path.exists(load_path) and not os.path.exists(load_path+".zip"):
            print("Failed to find", load_path)
        print("load path", load_path)
        agent = load_agent(args.algo, load_path, env, args.device, deterministic=False)

    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)

    cv2.namedWindow(args.env, cv2.WINDOW_NORMAL)
    keys = {27:'escape', 8:'backspace', 9:'tab', 13:'enter', 32:' ', 81:'left', 82:'up', 83:'right', 84:'down'}
    # assert ((len(env.observation_space.shape) == 3) and args.agent_view) or (not args.agent_view), "agent_view is only for RGB image observations"  

    # Eval
    images = []
    rewards, success, episode, t = 0, 0, 0, 0
    lstm_states = None
    lstm_episode_starts = np.ones((1,), dtype=bool)

    obs, _ = env.reset(seed=args.seed)
    while True:
        # plt.pause(0.5)
        key = cv2.waitKey(int(not args.enjoy_step_by_step))
        if key == 27 or (args.save_video and episode>=5): 
            cv2.destroyAllWindows()
            break
        
        img = obs
        if not args.agent_view: 
            img = env.render()
        if args.agent_view or  args.render_mode == "rgb_array":
            # img = cv2.resize(img, dsize=(512,int(512*(img.shape[0]/img.shape[1]))), interpolation=cv2.INTER_AREA)
            if args.save_video: # and t%5==0: # Uncomment to reduce number of frames and hence gif size
                images.append(img)
            img = transform_rgb_bgr(img)
            cv2.imshow(args.env, img) 
        
        if args.algo=="RecurrentPPO":
            action, lstm_states = agent(obs, lstm_states=lstm_states, lstm_episode_starts=lstm_episode_starts)
        else:
            action = agent(obs)

        obs, reward, done, truncate, info = env.step(action)
        
        rewards += reward
        t += 1
        lstm_episode_starts = done
        if done or truncate or key == 8:
            success += reward>0
            episode += 1
            t = 0
            print("episode",episode,"success",reward>0,"success rate",success/episode,"rewards",rewards/episode)

            img = obs
            if not args.agent_view: 
                img = env.render()
            if args.agent_view or args.render_mode == "rgb_array":
                # img = cv2.resize(img, dsize=(512,int(512*(img.shape[0]/img.shape[1]))), interpolation=cv2.INTER_AREA)
                if args.save_video:
                    images.append(img)
                img = transform_rgb_bgr(img)
                cv2.imshow(args.env, img) 
            
            lstm_states = None
            lstm_episode_starts = np.ones((1,), dtype=bool)
            obs, _ = env.reset(seed=None if args.seed==None else args.seed+episode)
    if args.save_video:
        print(name)
        # imageio.mimsave(f"images/{name}.mp4", images, fps=60)
        imageio.mimsave(f"images/{name}.gif", images, loop=0, fps=60)