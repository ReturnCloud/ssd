import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_agents, agent_id, episode_length, n_rollout_threads, obs_space, action_space,
                 recurrent_hidden_state_size):

        self.agent_id = agent_id
        obs_shape = obs_space.shape
        if len(obs_shape) == 3:
            self.share_obs = torch.zeros(episode_length + 1, n_rollout_threads, num_agents*obs_shape[0],obs_shape[1], obs_shape[2])
        elif len(obs_shape) == 1:
            self.share_obs = torch.zeros(episode_length + 1, n_rollout_threads, obs_shape[0] * num_agents)
        else:
            raise NotImplementedError
        self.obs = torch.zeros(episode_length + 1, n_rollout_threads, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            episode_length + 1, n_rollout_threads, recurrent_hidden_state_size)
        self.recurrent_c_states = torch.zeros(
            episode_length + 1, n_rollout_threads, recurrent_hidden_state_size)
        self.recurrent_hidden_states_critic = torch.zeros(
        episode_length + 1, n_rollout_threads, recurrent_hidden_state_size)
        self.recurrent_c_states_critic = torch.zeros(
        episode_length + 1, n_rollout_threads, recurrent_hidden_state_size)
        self.rewards = torch.zeros(episode_length, n_rollout_threads, 1)
        self.value_preds = torch.zeros(episode_length + 1, n_rollout_threads, 1)
        self.returns = torch.zeros(episode_length + 1, n_rollout_threads, 1)
        self.action_log_probs = torch.zeros(episode_length, n_rollout_threads, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(episode_length, n_rollout_threads, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(episode_length + 1, n_rollout_threads, 1)

        # Masks that indicate whether it's a true terminal state or time limit end state
        self.bad_masks = torch.ones(episode_length + 1, n_rollout_threads, 1)

        self.episode_length = episode_length
        self.step = 0

    def to(self, device):
        self.share_obs = self.share_obs.to(device)
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.recurrent_hidden_states_critic = self.recurrent_hidden_states_critic.to(device)
        self.recurrent_c_states = self.recurrent_c_states.to(device)
        self.recurrent_c_states_critic = self.recurrent_c_states_critic.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, share_obs, obs, recurrent_hidden_states, recurrent_hidden_states_critic, recurrent_c_states, recurrent_c_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks):
        self.share_obs[self.step + 1].copy_(share_obs)
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.recurrent_hidden_states_critic[self.step + 1].copy_(recurrent_hidden_states_critic)
        self.recurrent_c_states[self.step + 1].copy_(recurrent_c_states)
        self.recurrent_c_states_critic[self.step + 1].copy_(recurrent_c_states_critic)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.recurrent_hidden_states_critic[0].copy_(self.recurrent_hidden_states_critic[-1])
        self.recurrent_c_states[0].copy_(self.recurrent_c_states[-1])
        self.recurrent_c_states_critic[0].copy_(self.recurrent_c_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        popart=None):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    if popart:
                        # step + 1
                        delta = self.rewards[step] + gamma * popart.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                                - popart.denormalize(self.value_preds[step])
                        gae = delta + gamma * gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step+1] - self.value_preds[step]
                        gae = delta + gamma * gae_lambda * self.masks[step+1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                if popart:
                    self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]) * \
                        self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * popart.denormalize(self.value_preds[step])
                else:
                    self.returns[-1] = next_value
                    for step in reversed(range(self.rewards.size(0))):
                        self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]) \
                            * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    if popart:
                        delta = self.rewards[step] + gamma * popart.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                                - popart.denormalize(self.value_preds[step])
                        gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + popart.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step+1] - self.value_preds[step]
                        gae = delta + gamma * gae_lambda * self.masks[step+1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.size()[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = self.share_obs[:-1].view(-1, *self.share_obs.size()[2:])[indices]
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            recurrent_hidden_states_critic_batch = self.recurrent_hidden_states_critic[:-1].view(
                -1, self.recurrent_hidden_states_critic.size(-1))[indices]
            recurrent_c_states_batch = self.recurrent_c_states[:-1].view(
                -1, self.recurrent_c_states.size(-1))[indices]
            recurrent_c_states_critic_batch = self.recurrent_c_states_critic[:-1].view(
                -1, self.recurrent_c_states_critic.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, recurrent_c_states_batch, recurrent_c_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.size(1)
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads)
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            recurrent_c_states_batch = []
            recurrent_c_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                recurrent_hidden_states_critic_batch.append(self.recurrent_hidden_states_critic[0:1, ind])
                recurrent_c_states_batch.append(self.recurrent_c_states[0:1, ind])
                recurrent_c_states_critic_batch.append(self.recurrent_c_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            share_obs_batch = torch.stack(share_obs_batch, 1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(recurrent_hidden_states_critic_batch, 1).view(N, -1)
            recurrent_c_states_batch = torch.stack(recurrent_c_states_batch, 1).view(N, -1)
            recurrent_c_states_critic_batch = torch.stack(recurrent_c_states_critic_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            share_obs_batch = _flatten_helper(T, N, share_obs_batch)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)


            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, recurrent_c_states_batch, recurrent_c_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.size()[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length #[C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(data_chunks)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            recurrent_c_states_batch = []
            recurrent_c_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(torch.transpose(self.share_obs[:-1],0,1).reshape(-1, *self.share_obs.size()[2:])[ind:ind+data_chunk_length])
                obs_batch.append(torch.transpose(self.obs[:-1],0,1).reshape(-1, *self.obs.size()[2:])[ind:ind+data_chunk_length])
                actions_batch.append(torch.transpose(self.actions,0,1).reshape(-1, self.actions.size(-1))[ind:ind+data_chunk_length])
                value_preds_batch.append(torch.transpose(self.value_preds[:-1],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                return_batch.append(torch.transpose(self.returns[:-1],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                masks_batch.append(torch.transpose(self.masks[:-1],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(torch.transpose(self.action_log_probs,0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                adv_targ.append(torch.transpose(advantages,0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                recurrent_hidden_states_batch.append(torch.transpose(self.recurrent_hidden_states[:-1],0,1).reshape(
                      -1, self.recurrent_hidden_states.size(-1))[ind])
                recurrent_hidden_states_critic_batch.append(torch.transpose(self.recurrent_hidden_states_critic[:-1],0,1).reshape(
                      -1, self.recurrent_hidden_states_critic.size(-1))[ind])
                recurrent_c_states_batch.append(torch.transpose(self.recurrent_c_states[:-1],0,1).reshape(
                      -1, self.recurrent_c_states.size(-1))[ind])
                recurrent_c_states_critic_batch.append(torch.transpose(self.recurrent_c_states_critic[:-1],0,1).reshape(
                      -1, self.recurrent_c_states_critic.size(-1))[ind])

            L, N =  data_chunk_length, mini_batch_size

            # These are all tensors of size (L, N, Dim)
            share_obs_batch = torch.stack(share_obs_batch)
            obs_batch = torch.stack(obs_batch)

            actions_batch = torch.stack(actions_batch)
            value_preds_batch = torch.stack(value_preds_batch)
            return_batch = torch.stack(return_batch)
            masks_batch = torch.stack(masks_batch)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch)
            adv_targ = torch.stack(adv_targ)

            # States is just a (N, -1) tensor

            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(recurrent_hidden_states_critic_batch).view(N, -1)
            recurrent_c_states_batch = torch.stack(recurrent_c_states_batch).view(N, -1)
            recurrent_c_states_critic_batch = torch.stack(recurrent_c_states_critic_batch).view(N, -1)

            # Flatten the (L, N, ...) tensors to (L * N, ...)
            share_obs_batch = _flatten_helper(L, N, share_obs_batch)
            obs_batch = _flatten_helper(L, N, obs_batch)
            actions_batch = _flatten_helper(L, N, actions_batch)
            value_preds_batch = _flatten_helper(L, N, value_preds_batch)
            return_batch = _flatten_helper(L, N, return_batch)
            masks_batch = _flatten_helper(L, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(L, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(L, N, adv_targ)

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, recurrent_c_states_batch, recurrent_c_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

