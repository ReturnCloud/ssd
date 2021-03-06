import copy
import glob
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from envs.harvest import HarvestEnv
from envs.cleanup import CleanupEnv
from algo.ppo import PPO
from algo.model import Policy
from algo.storage import RolloutStorage
from config import get_config
from envs.vec_env import SubprocVecEnv #, DummyVecEnv
from algo.algo_utils import update_linear_schedule
from utils import *
import matplotlib.pyplot as plt
import time

def main():
    args = get_config()

    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    frames_dir = f'./experiments/cleanup_{args.n_rollout_threads}/{args.num_agents}_none/video'

    # env
    if args.env_name == "cleanup":
        env = CleanupEnv(num_agents=args.num_agents)
    elif args.env_name == "haevest":
        env = HarvestEnv(num_agents=args.num_agents)
    else:
        print("Can not support the " + args.env_name + "environment." )
        raise NotImplementedError

    #Policy network
    actor_critic = []
    for i in range(args.num_agents):
        ckpt = torch.load(f'./experiments/cleanup_{args.n_rollout_threads}/{args.num_agents}_none/model/agent_{i}.pth')
        ac = Policy(env.observation_space[0],
                    env.action_space[0],
                    num_agents = args.num_agents,
                    base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                'recurrent': args.recurrent_policy,
                                'hidden_size': args.hidden_size})
        ac.load_state_dict(ckpt)
        ac.to(device)

        actor_critic.append(ac)
    for episode in range(1):
        state = env.reset()
        state = np.array([state])

        share_obs = []
        obs = []
        recurrent_hidden_statess = []
        recurrent_hidden_statess_critic = []
        recurrent_c_statess = []
        recurrent_c_statess_critic = []
        masks = []
        rewards = [0 for _ in range(args.num_agents+1)]

        # rollout
        for i in range(args.num_agents):
            if len(env.observation_space[0].shape) == 1:
                share_obs.append((torch.tensor(state.reshape(1, -1),dtype=torch.float32)).to(device))
                obs.append((torch.tensor(state[:,i,:],dtype=torch.float32)).to(device))
            else:
                cur_share_obs = np.concatenate([state[:,i,:,:,:] for i in range(args.num_agents)], axis=1)
                share_obs.append(torch.tensor(cur_share_obs).to(device))
                obs.append((torch.tensor(state[:,i,:,:,:],dtype=torch.float32)).to(device))
            recurrent_hidden_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))
            recurrent_hidden_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))
            recurrent_c_statess.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))
            recurrent_c_statess_critic.append(torch.zeros(1, actor_critic[i].recurrent_hidden_state_size).to(device))
            masks.append(torch.ones(1,1).to(device))

        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        for step in range(args.episode_length):
            # print("step %i of %i" % (step, args.episode_length))
            # Sample actions
            img = env.render()
            # print (img.shape, img.mean())
            ax.cla()
            ax.imshow(img)
            ax.set_title('step ' + str(step))
            time.sleep(0.1)
            plt.imsave(f'{frames_dir}/img_{episode*args.episode_length+step}.png', img)

            actions_env = []
            for i in range(args.num_agents):
                # one_hot_action = np.zeros(env.action_space[0].n)
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic,recurrent_c_states, recurrent_c_states_critic = actor_critic[i].act(share_obs[i], obs[i], recurrent_hidden_statess[i], recurrent_hidden_statess_critic[i], recurrent_c_statess[i], recurrent_c_statess_critic[i], masks[i])
                recurrent_hidden_statess[i].copy_(recurrent_hidden_states)
                recurrent_hidden_statess_critic[i].copy_(recurrent_hidden_states_critic)
                recurrent_c_statess[i].copy_(recurrent_c_states)
                recurrent_c_statess_critic[i].copy_(recurrent_c_states_critic)
                actions_env.append(action)
                # one_hot_action[action] = 1
                # one_hot_actions.append(one_hot_action)

            # Obser reward and next obs
            state, reward, done, infos = env.step(actions_env)
            if any(done):
                break

            for i in range(args.num_agents):
                # print("Reward of agent%i: " %i + str(reward[i]))
                rewards[i] += reward[i]
            state = np.array([state])
            rewards[-1] = sum(rewards[:-1])
            print (f'reward {reward}, spawn: {env.current_apple_spawn_prob, env.current_waste_spawn_prob}')
            for i in range(args.num_agents):
                if len(env.observation_space[0].shape) == 1:
                    share_obs[i].copy_(torch.tensor(state.reshape(1, -1),dtype=torch.float32))
                    obs[i].copy_(torch.tensor(state[:,i,:],dtype=torch.float32))
        print (f'episode {episode}, {rewards}')
    os.system(
        'ffmpeg -r 5 -start_number 0 -i ' +
        f'{frames_dir}/img_%d.png -c:v libx264 -pix_fmt yuv420p ' +
        f'{frames_dir}/video.mp4')
    for name in os.listdir(frames_dir):
        if '.png' in name:
            os.remove(f'{frames_dir}/{name}')

if __name__ == "__main__":
    main()