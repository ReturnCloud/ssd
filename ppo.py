Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore

@ReturnCloud
Learn Git and GitHub without any code!
Using the Hello World guide, you’ll start a branch, write comments, and open a pull request.


ReturnCloud
/
sir
Private
1
00
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Settings
sir/th/ppo.py /
@ReturnCloud
ReturnCloud tmp 8803
Latest commit b1c50d7 on 25 Dec 2020
 History
 1 contributor
242 lines (215 sloc)  10.4 KB

import csv
import numpy as np
import gym
import torch
import torch.nn as nn
# from distribution import make_prob_dist, DiagGaussion
from torch.distributions.normal import Normal
from ppo_model import *
import time
from torch import distributions as td


SAVE_FREQ = 100
EVAL_FREQ = 1

class PPO():
    # need: env, optimizer
    def __init__(self, policy, buffer, device, lr, n_total_interact, n_step, n_env, ppg_hps, sil=None):
        self.policy = policy.to(device)
        self.buffer = buffer
        self.device = device
        self.lr = lr
        self.n_pol_epoch = ppg_hps['n_pol_epoch']
        self.n_pol_update = ppg_hps['n_pol_update']
        self.n_pol_batch = ppg_hps['n_pol_batch']
        self.n_aux_epoch = ppg_hps['n_aux_epoch']
        self.n_aux_update = ppg_hps['n_aux_update']
        self.n_aux_batch = ppg_hps['n_aux_batch']
        self.n_total_interact = n_total_interact
        self.n_step = n_step
        self.n_env = n_env
        self.sil = sil
        self.pol_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.aux_opt = torch.optim.Adam(policy.parameters(), lr=lr)


    def compute_loss(self, data, ratio_clip, vf_clip, vf_coef, ent_coef):
        obs, rets, acts, vals, old_logps, advs = data

        obs = torch.from_numpy(obs).to(self.device).float()
        acts = torch.from_numpy(acts).to(self.device).float()
        advs = torch.from_numpy(advs).to(self.device).float()
        old_logps = torch.from_numpy(old_logps).to(self.device)
        rets = torch.from_numpy(rets).to(self.device)

        new_vals, new_logps, ent_loss = self.policy.eval_action(
            obs, acts)
        ratio = torch.exp(new_logps - old_logps)
        clip_advs = torch.clamp(ratio, 1-ratio_clip, 1+ratio_clip) * advs
        pi_loss = -(torch.min(ratio * advs, clip_advs)).mean()

        clip_vals = new_vals
        if vf_clip:
            clip_vals = vals + (new_vals - vals).clamp(-vf_clip, vf_clip)
        vf_loss1 = (new_vals-rets) ** 2
        vf_loss2 = (clip_vals-rets) ** 2
        vf_loss = 0.5*(torch.max(vf_loss1, vf_loss2)).mean()

        loss = pi_loss + vf_loss*vf_coef - ent_loss*ent_coef

        approx_kl = 0.5*((old_logps - new_logps)**2).mean().item()
        return loss, pi_loss, vf_loss, ent_loss, approx_kl

    def gen_trajectory(self, env, gae_hps, epoch):
        gamma, lam = gae_hps.values()
        for t in range(self.n_step):
            step = self.n_step*epoch + t
            act, logp, val = self.policy.step(self.buffer.obs[t], self.device)
            clipped_act = act
            if isinstance(env.action_space, gym.spaces.Box):
                clipped_act = np.clip(act, env.action_space.low, env.action_space.high)
            new_obs, rwd, done, info = env.step(clipped_act)
            # print (act.shape, logp.shape, val.shape, new_obs.shape, rwd.shape, done.shape)
            # print (t, 'done:', done, 'rwd:', rwd, 'info:', info)
            self.buffer.store(step, new_obs, rwd, act, val, done, logp)
            if self.sil:
                self.sil.step(new_obs, obs, rwd, act, val, done, logp, info, gamma, lam)
            obs = new_obs

        new_act, new_logp, new_val = self.policy.step(self.buffer.obs[-1], self.device)    # final
        self.buffer.estimate_adv(new_val, gamma, lam)

    def pol_train(self, epoch, pol_hps, writer):
        inds = np.arange(self.n_step * self.n_env)
        batch_size = self.n_step * self.n_env // self.n_pol_batch
        # print ('in pol train, batch_size:', batch_size)
        ratio_clip, vf_clip, vf_coef, ent_coef, grad_clip = pol_hps.values()
        for update in range(self.n_pol_update):
            tloss, tpi_loss, tvf_loss, tent_loss, tapp_kl = 0, 0, 0, 0, 0
            np.random.shuffle(inds)
            for start in range(0, self.n_step * self.n_env, batch_size):
                end = min(start + batch_size,
                            self.n_step * self.n_env)
                indices = inds[start:end]
                sample = self.buffer.get(indices)

                loss, pi_loss, vf_loss, ent_loss, approx_kl = self.compute_loss(sample, ratio_clip, vf_clip, vf_coef, ent_coef)
                self.pol_opt.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
                self.pol_opt.step()
                tloss += loss
                tpi_loss += pi_loss
                tvf_loss += vf_loss
                tent_loss += ent_loss
                tapp_kl += approx_kl
            writer.add_scalar('pi_loss', tpi_loss / self.n_pol_batch, epoch*self.n_pol_update + update)
            writer.add_scalar('vf_loss', tvf_loss / self.n_pol_batch, epoch*self.n_pol_update + update)
            writer.add_scalar('ent_loss', tent_loss / self.n_pol_batch, epoch*self.n_pol_update + update)
            writer.add_scalar('approx_kl', tapp_kl / self.n_pol_batch, epoch*self.n_pol_update + update)
        if self.sil:
            self.sil.train_sil()

    def compute_aux_loss(self, data, aux_hps):
        obs, ret, oldpd = data
        kl_coef = aux_hps['kl_coef']
        # pi_hidden, val = self.policy.aux_step(obs, self.device)
        # th_pi = torch.tensor(pi_hidden).to(self.device)
        # pd = self.policy.dist(th_pi)
        pd, val_dict = self.policy.aux_step(obs, self.device)
        aux_kl = td.kl_divergence(oldpd, pd).mean()
        aux_vf = 0
        for k in val_dict.keys():
            v = val_dict[k]
            aux_vf += torch.tensor(0.5 * ((v-ret)**2).mean())
        aux_loss = aux_vf + kl_coef*aux_kl
        return aux_loss, aux_vf, aux_kl

    def get_aux_sample(self, obs, ret, oldpd, ids):
        batch_obs = obs[ids,:]
        batch_ret = ret[ids,:]
        # batch_pi = torch.tensor(pi_hidden[ids,:]).to(self.device)
        # batch_pd = self.policy.dist.forward(batch_pi)
        batch_pd = oldpd.get_slices(ids)
        return batch_obs, batch_ret, batch_pd

    def aux_train(self, env, seg_obs, seg_ret, seg_pd, aux_hps, epoch, writer):
        inds = np.arange(seg_obs.shape[0])
        # print (seg_obs.shape)
        batch_size = seg_obs.shape[0] // self.n_aux_batch
        for i in range(self.n_aux_update):
            tloss, tvf_loss, tkl_loss = 0, 0, 0
            np.random.shuffle(inds)
            for start in range(0, seg_obs.shape[0], batch_size):
                end = min(start+batch_size, seg_obs.shape[0])
                batch_ids = inds[start:end]
                data = self.get_aux_sample(seg_obs, seg_ret, seg_pd, batch_ids)
                aux_loss, aux_vf, aux_kl = self.compute_aux_loss(data, aux_hps)
                self.aux_opt.zero_grad()
                # torch.autograd.set_detect_anomaly(True)
                aux_loss.backward()
                self.aux_opt.step()
                # tvf_loss += aux_vf
                # tkl_loss += aux_kl
            # writer.add_scalar('aux_vf', tvf_loss/self.n_aux_batch, epoch+i)
            # writer.add_scalar('aux_kl', tkl_loss/self.n_aux_batch, epoch+i)
    def pre_aux_out(self, seg_obs, seg_ret):
        def cat_dists(dists):
            means, stds = [], []
            for d in dists:
                means.append(d.loc.detach())
                stds.append(d.scale.detach())
            mean = torch.cat(means, dim=0)
            std = torch.cat(stds, dim=0)
            return FixedNormal(mean, std)
        # oldpds, oldvs_dict = [], {'true':[], 'aux':[]}
        oldpds = []
        inds = np.arange(seg_obs.shape[0])
        batch_size = seg_obs.shape[0] // self.n_aux_batch
        # print ('in aux train, batch size:', batch_size, 'seg_obs len:', seg_obs.shape[0])
        for start in range(0, seg_obs.shape[0], batch_size):
            end = min(start+batch_size, seg_obs.shape[0])
            batch_ids = inds[start:end]
            # print (start, end)
            with torch.no_grad():
                boldpd, boldv_dict = self.policy.aux_step(seg_obs[batch_ids,:], self.device)
            oldpds.append(boldpd)
            # for k in boldv_dict:
            #     oldvs_dict[k].append(boldv_dict[k])
        oldpd = cat_dists(oldpds)
        # oldv_dict = {}
        # for k in oldvs_dict:
        #     oldv_dict[k] = np.concatenate(oldvs_dict[k])
        return oldpd #, oldv_dict


    def train(self, env, gae_hps, pol_hps, aux_hps, save_path, writer=None):
        obs = env.reset()
        self.buffer.obs[0] = obs.copy()
        epoch, n_interact = 0, 0
        segs = {'obs':[], 'ret':[]}

        while True:
            if epoch % EVAL_FREQ == 0:
                begin = time.time()

            for i in range(self.n_pol_epoch):
                if (epoch + 1) % EVAL_FREQ == 0:
                    begin = time.time()
                self.gen_trajectory(env, gae_hps, epoch)
                self.pol_train(epoch, pol_hps, writer)
                segs['obs'].append(self.buffer.obs[:-1])
                segs['ret'].append(self.buffer.returns[:-1])
                n_interact += self.n_step * self.n_env
                s_rwd, n_rwd = self.buffer.log_rwd()
                writer.add_scalar('reward', s_rwd / n_rwd, epoch)
                self.buffer.clear()
                torch.cuda.empty_cache()

                if (epoch + 1) % EVAL_FREQ == 0:
                    finish = time.time()
                    print(f'{epoch+1} epoch, time cost {(finish-begin):.2f}, mean reward {(s_rwd/n_rwd):.3f}')
                    # self.policy.debug()
                if (epoch + 1) % SAVE_FREQ == 0:
                    ckpt_path = f'{save_path}/{epoch+1}.pth'
                    torch.save(self.policy.state_dict(), ckpt_path)
                epoch += 1

            if n_interact >= self.n_total_interact:
                break
            if self.n_aux_epoch > 0:
                seg_obs = np.concatenate(segs['obs']).reshape(-1,segs['obs'][0].shape[-1])
                seg_ret = np.concatenate(segs['ret']).reshape(-1,segs['ret'][0].shape[-1])
                oldpd = self.pre_aux_out(seg_obs, seg_ret)
                for i in range(self.n_aux_epoch):
                    self.aux_train(env, seg_obs, seg_ret, oldpd, aux_hps, epoch, writer)
                torch.cuda.empty_cache()
            segs = {'obs':[], 'ret':[]}



    def buffer_debug(self):
        f = open('self_debug.csv', 'w')
        f_csv = csv.writer(f)
        f_csv.writerows(self.buffer.rewards.transpose(1,0,2))
        f_csv.writerows(self.buffer.dones.transpose(1, 0, 2))
        f_csv.writerows(self.buffer.returns.transpose(1, 0, 2))
        f_csv.writerows(self.buffer.values.transpose(1, 0, 2))
        f_csv.writerows([self.buffer.np_advs])

