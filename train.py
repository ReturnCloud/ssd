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
from gpu_memory_log import *

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "cleanup":
                env = CleanupEnv(num_agents=args.num_agents)
            elif args.env_name == "harvest":
                env = HarvestEnv(num_agents=args.num_agents)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    if args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])

def main():
    args = get_config()

    # ----------------- seed ------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # cuda
    if args.cuda>=0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    # ----------------- prepare for env, network, algo, buffer, logger ------------------
    attr = f'./experiments/{args.env_name}_{args.n_rollout_threads}/{args.num_agents}_'
    if args.lstm:
        attr += 'lstm'
    elif args.recurrent_policy:
        attr += 'recurrent'
    elif args.naive_recurrent_policy:
        attr += 'naive_recurrent'
    else:
        attr += 'none'

    log_dir = f'{attr}/log'
    ckpt_dir = f'{attr}/model'
    reset_dir(log_dir)
    mk_dir(ckpt_dir)
    logger = SummaryWriter(log_dir)
    # logger = None
    envs = make_parallel_env(args)
    actor_critic = []
    if args.share_policy:
        ac = Policy(envs.observation_space[0],
                    envs.action_space[0],
                    num_agents = args.num_agents,
                    base_kwargs={'lstm': args.lstm,
                                 'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size})
        ac.to(device)
        for agent_id in range(args.num_agents):
            actor_critic.append(ac)
    else:
        for agent_id in range(args.num_agents):
            ac = Policy(envs.observation_space[0],
                      envs.action_space[0],
                      num_agents = args.num_agents,
                      base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                   'recurrent': args.recurrent_policy,
                                   'hidden_size': args.hidden_size})
            if args.load:
                ckpt = torch.load(f'./experiments/cleanup_{args.n_rollout_threads}/{args.num_agents}_none/model/agent_{agent_id}.pth')
                ac.load_state_dict(ckpt)
                print ('load previous model')

            ac.to(device)
            actor_critic.append(ac)
    # gpu_memory_log()
    agents = []
    rollouts = []
    for agent_id in range(args.num_agents):
        agent = PPO(actor_critic[agent_id],
                   agent_id,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   lr=args.lr,
                   eps=args.eps,
                   max_grad_norm=args.max_grad_norm,
                   use_clipped_value_loss=args.use_clipped_value_loss)
        ro = RolloutStorage(args.num_agents,
                            agent_id,
                            args.episode_length,
                            args.n_rollout_threads,
                            envs.observation_space[agent_id],
                            envs.action_space[agent_id],
                            actor_critic[agent_id].recurrent_hidden_state_size)
        agents.append(agent)
        rollouts.append(ro)

    # ----------------- reset env ------------------
    obs = envs.reset() # (n_thread, n_agent, c, h, w)
    cur_share_obs = np.concatenate([obs[:,i,:,:,:] for i in range(args.num_agents)], axis=1)
    for i in range(args.num_agents):
        rollouts[i].share_obs[0].copy_(torch.tensor(cur_share_obs))
        rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:,:,:]))
        rollouts[i].recurrent_hidden_states.zero_()
        rollouts[i].recurrent_hidden_states_critic.zero_()
        rollouts[i].recurrent_c_states.zero_()
        rollouts[i].recurrent_c_states_critic.zero_()
        rollouts[i].to(device)

    # ----------------- run ------------------
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    # episodes=1
    all_episode = 0

    for episode in range(episodes):
        action_fire, action_clean = {f'agent_{i}': 0 for i in range(args.num_agents)}, {f'agent_{i}': 0 for i in range(args.num_agents)}
        spawn_info = {'apple':0, 'waste':0}
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            for i in range(args.num_agents):
                update_linear_schedule(agents[i].optimizer, episode, episodes, args.lr)

        for step in range(args.episode_length):
            # Sample actions
            values = []
            actions= []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            recurrent_c_statess = []
            recurrent_c_statess_critic = []

            with torch.no_grad():
                for i in range(args.num_agents):
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic,\
                        recurrent_c_states, recurrent_c_states_critic =\
                            actor_critic[i].act(rollouts[i].share_obs[step],
                                                        rollouts[i].obs[step],
                                                        rollouts[i].recurrent_hidden_states[step],
                                                        rollouts[i].recurrent_hidden_states_critic[step],
                                                        rollouts[i].recurrent_c_states[step],
                                                        rollouts[i].recurrent_c_states_critic[step],
                                                        rollouts[i].masks[step])


                    values.append(value)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)
                    recurrent_hidden_statess.append(recurrent_hidden_states)
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic)
                    recurrent_c_statess.append(recurrent_c_states)
                    recurrent_c_statess_critic.append(recurrent_c_states_critic)

            # rearrange action
            actions_env = []
            for i in range(args.n_rollout_threads):
                action_env = []
                for k in range(args.num_agents):
                    action_env.append(int(actions[k][i]))
                    if actions[k][i] == 7:
                        action_fire[f'agent_{k}'] += 1 / (args.episode_length*args.n_rollout_threads)
                    if actions[k][i] == 8:
                        action_clean[f'agent_{k}'] += 1 / (args.episode_length*args.n_rollout_threads)
                actions_env.append(action_env)
            # print (f'---------------------------------{step}-----------------------------')
            # gpu_memory_log()
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(actions_env)
            cur_share_obs = np.concatenate([obs[:,i,:,:,:] for i in range(args.num_agents)], axis=1)
            for i in range(args.n_rollout_threads):
                for k in spawn_info.keys():
                    spawn_info[k] += infos[i][k] / (args.n_rollout_threads*args.episode_length)
            # insert data in buffer, if done then clean the history of observations.
            masks, bad_masks = [], []
            for i in range(args.num_agents):
                mask, bad_mask = [], []
                for done_ in done:
                    if done_[i]:
                        mask.append([0.0])
                        bad_mask.append([1.0])
                    else:
                        mask.append([1.0])
                        bad_mask.append([1.0])
                masks.append(torch.FloatTensor(mask))
                bad_masks.append(torch.FloatTensor(bad_mask))

            for i in range(args.num_agents):
                rollouts[i].insert(torch.tensor(cur_share_obs),
                                        torch.tensor(obs[:,i,:,:,:]),
                                        recurrent_hidden_statess[i],
                                        recurrent_hidden_statess_critic[i],
                                        recurrent_c_statess[i],
                                        recurrent_c_statess_critic[i],
                                        actions[i],
                                        action_log_probs[i],
                                        values[i],
                                        torch.tensor(reward[:, i].reshape(-1,1)),
                                        masks[i],
                                        bad_masks[i])

        with torch.no_grad():
            next_values = []
            for i in range(args.num_agents):
                next_value = actor_critic[i].get_value(rollouts[i].share_obs[-1],
                                                       rollouts[i].obs[-1],
                                                       rollouts[i].recurrent_hidden_states[-1],
                                                       rollouts[i].recurrent_hidden_states_critic[-1],
                                                       rollouts[i].recurrent_c_states[-1],
                                                       rollouts[i].recurrent_c_states_critic[-1],
                                                       rollouts[i].masks[-1]).detach()
                # print (f'---------------------------------{step} {i}-----------------------------')
                # gpu_memory_log()
                next_values.append(next_value)

        for i in range(args.num_agents):
            rollouts[i].compute_returns(next_values[i],
                                        args.use_gae,
                                        args.gamma,
                                        args.gae_lambda,
                                        args.use_proper_time_limits)

         # ----------------- update and log ------------------
        value_losses, action_losses, dist_entropies, rwds = {}, {}, {}, {'all': 0}
        for i in range(args.num_agents):
            value_loss, action_loss, dist_entropy, rwd = agents[i].update(rollouts[i])
            rwds[f'agent_{i}'] = rwd
            value_losses[f'agent_{i}'] = value_loss
            action_losses[f'agent_{i}'] = action_loss
            dist_entropies[f'agent_{i}'] = dist_entropy
        for k in range(args.num_agents):
            rwds['all'] += rwds[f'agent_{k}']
        logger.add_scalars('value_loss', value_losses, episode)
        logger.add_scalars('action_loss', action_losses, episode)
        logger.add_scalars('dist_entropy', dist_entropies, episode)
        logger.add_scalars('rwd', rwds, episode)
        logger.add_scalars('fire', action_fire, episode)
        logger.add_scalars('clean', action_clean, episode)
        logger.add_scalars('spawn', spawn_info, episode)

        # ----------------- clean the buffer and reset ------------------
        obs = envs.reset()
        cur_share_obs = np.concatenate([obs[:,i,:,:,:] for i in range(args.num_agents)], axis=1)
        for i in range(args.num_agents):
            rollouts[i].share_obs[0].copy_(torch.tensor(cur_share_obs))
            rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
            rollouts[i].recurrent_hidden_states.zero_()
            rollouts[i].recurrent_hidden_states_critic.zero_()
            rollouts[i].recurrent_c_states.zero_()
            rollouts[i].recurrent_c_states_critic.zero_()
            rollouts[i].masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
            rollouts[i].bad_masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
            rollouts[i].to(device)
        if (episode % args.save_interval == 0 or episode == episodes - 1):
            print (f'save the model of episode {episode}, {rwds}, env {spawn_info}')
            for i in range(args.num_agents):
                torch.save(actor_critic[i].state_dict(), f'{ckpt_dir}/agent_{i}.pth')


if __name__ == '__main__':
    main()