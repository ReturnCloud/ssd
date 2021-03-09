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
from mappo.r_mappo.r_mappo import R_MAPPO
from mappo.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from algo.storage import RolloutStorage
from config import get_config
from envs.vec_env import SubprocVecEnv #, DummyVecEnv
from algo.algo_utils import update_linear_schedule
from utils import *
from gpu_memory_log import *
#!/usr/bin/env python
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

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



def main(args):
    all_args = get_config()

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError


    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()




if __name__ == "__main__":
    main()