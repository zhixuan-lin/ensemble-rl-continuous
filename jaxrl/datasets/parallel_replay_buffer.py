from typing import Optional

import os
import gym
import pickle
import numpy as np
from pathlib import Path

from jaxrl.datasets.dataset import Dataset, Batch


class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, num_buffers: int):
        self.observations = np.empty((num_buffers, capacity, observation_space.shape[-1]),
                                     dtype=observation_space.dtype)
        self.actions = np.empty((num_buffers, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_buffers, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_buffers, capacity, ), dtype=np.float32)
        self.dones_float = np.empty((num_buffers, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_buffers, capacity, observation_space.shape[-1]),
                                          dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.num_buffers = num_buffers

    @property
    def data_keys(self):
        return ['observations', 'actions', 'rewards', 'masks', 'dones_float', 'next_observations']

    @property
    def state_keys(self):
        return ['size', 'insert_index', 'capacity', 'num_buffers']

    def save(self, path: str):
        path = Path(path)
        assert path.is_dir()
        data = {key: getattr(self, key) for key in self.data_keys}
        np.savez(path / 'buffer_data.npz',
                 **data)
        state = {key: getattr(self, key) for key in self.state_keys}
        with (path / 'buffer_state.pickle').open('wb') as f: 
            pickle.dump(state, f)

    def load(self, path: str):
        path = Path(path)
        assert path.is_dir()
        with np.load(path / 'buffer_data.npz') as data:
            assert sorted(data.keys()) == sorted(self.data_keys)
            for key, value in data.items():
                setattr(self, key, value)
        with (path / 'buffer_state.pickle').open('rb') as f:
            state = pickle.load(f)
            assert sorted(state.keys()) == sorted(self.state_keys)
            for key, value in state.items():
                setattr(self, key, value)


    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel(self, batch_size: int) -> Batch:
        buffer_index = np.arange(self.num_buffers)[:, None]
        pos_index = np.random.randint(self.size, size=(self.num_buffers, batch_size))
        return Batch(observations=self.observations[buffer_index, pos_index],
                     actions=self.actions[buffer_index, pos_index],
                     rewards=self.rewards[buffer_index, pos_index],
                     masks=self.masks[buffer_index, pos_index],
                     next_observations=self.next_observations[buffer_index, pos_index])

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        buffer_index = np.arange(self.num_buffers)[:, None, None]
        pos_index = np.random.randint(self.size, size=(self.num_buffers, num_batches, batch_size))
        return Batch(observations=self.observations[buffer_index, pos_index],
                     actions=self.actions[buffer_index, pos_index],
                     rewards=self.rewards[buffer_index, pos_index],
                     masks=self.masks[buffer_index, pos_index],
                     next_observations=self.next_observations[buffer_index, pos_index])
