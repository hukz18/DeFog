import os
import gym
import sys
import torch
import d4rl_atari
import numpy as np
from dotmap import DotMap
from drop_fn import get_drop_fn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AtariBuffer:
    def __init__(self, env, dataset_type, context_len, stack_frame, drop_cfg, sample_type='traj_length', seed=0) -> None:
        self.dataset_type = dataset_type
        if dataset_type == 'expert-replay':
            # TODO: stack frame for expert replay
            raise NotImplementedError
        elif dataset_type in ['medium', 'expert', 'mixed']:
            # self.dataset, _= d3rlpy.datasets.get_dataset(f'{env_name.lower()}-{dataset_type}-v0')
            self.dataset = DotMap(env.get_dataset())
        self.rng = np.random.default_rng(seed)

        self.traj_sp = np.insert(np.where(self.dataset.terminals), 0, 0)
        self.dataset.terminals[-1] = True
        self.dataset.terminals = self.dataset.terminals.astype(np.bool)
        # self.dataset.observations is a list to save memory
        self.traj_ep = np.where(self.dataset.terminals)[0]
        self.traj_length = self.traj_ep - self.traj_sp
        if self.traj_length[-1] < context_len: # pad to avoid index out of bound when fetching the last trajecectory, this is safe as we mask out padded steps.
            for _ in range(context_len - self.traj_length[-1]):
                self.dataset.observations.append(np.zeros_like(self.dataset.observations[0]))
        self.traj_returns = np.add.reduceat(self.dataset.rewards, self.traj_sp)
        self.num_trajs = len(self.traj_sp)
        self.rewards_to_go = np.cumsum(self.traj_returns)[np.insert(np.cumsum(self.dataset.terminals), 0, 0)[:-1]] - np.cumsum(self.dataset.rewards)
        self.p_sample = np.ones(self.num_trajs) / self.num_trajs if sample_type == 'uniform' else self.traj_returns / \
            self.traj_returns.sum() if sample_type == 'traj_return' else self.traj_length / self.traj_length.sum()
        
        self.context_len = context_len
        self.stack_frame = stack_frame
        self.size = self.dataset.rewards.shape[0] - self.context_len * self.num_trajs
        self.drop_fn = get_drop_fn(drop_cfg, self.dataset.rewards.shape[0], self.traj_sp, self.rng)

    def sample(self, batch_size):
        selected_traj = self.rng.choice(np.arange(self.num_trajs), batch_size, replace=True, p=self.p_sample)
        selected_traj_sp = self.traj_sp[selected_traj]
        selected_offset = np.floor(self.rng.random(batch_size) * (self.traj_length[selected_traj] - self.context_len)).astype(np.int32).clip(min=0)
        selected_sp = selected_traj_sp + selected_offset
        selected_ep = (selected_sp + self.context_len).clip(max=self.traj_ep[selected_traj])

        # fill the index of those padded steps with -1, so that we can fetch the last step of the corresponding item
        selected_index = selected_sp[:, None] + np.arange(self.context_len)
        selected_index = np.where(selected_index < selected_ep[:, None], selected_index, -1)
        masks = selected_index >= 0
        timesteps = selected_offset[:, None] + np.arange(self.context_len)  # we don't care about the timestep for those padded steps
        
        # update and get drop mask
        self.drop_fn.step()
        dropsteps = self.drop_fn.get_dropsteps(selected_index)
        observation_index = selected_index - dropsteps

        states = torch.stack([torch.from_numpy(np.stack(self.dataset.observations[sp: sp+self.context_len])).to(dtype=torch.float32, device=device) for sp in selected_sp])
        actions = torch.as_tensor(self.dataset.actions[selected_index]).to(dtype=torch.int32, device=device)
        rewards_to_go = torch.as_tensor(self.rewards_to_go[observation_index, None]).to(dtype=torch.float32, device=device)
        timesteps = torch.as_tensor(timesteps).to(dtype=torch.int32, device=device)
        dropsteps = torch.as_tensor(dropsteps).to(dtype=torch.int32, device=device)


        return states, actions, rewards_to_go, timesteps, dropsteps, masks