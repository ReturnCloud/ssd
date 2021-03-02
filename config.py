import argparse

def get_config():
    parser = argparse.ArgumentParser(description='ssd')

    parser.add_argument('--env_name', default='clean_up')
    parser.add_argument('--n_rollout_threads', default=64)
    parser.add_argument('--n_training_threads', default=1)
    parser.add_argument('--cuda', action='store_false', default=True)
    parser.add_argument('--num_agents', default=1)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--cuda_deterministic', action='store_false', default=True)
    parser.add_argument('--episode_length', default=50)
    parser.add_argument('--num_env_steps', type=int, default=3000000)

    args = parser.parse_args()
    return args