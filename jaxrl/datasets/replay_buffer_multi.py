from typing import Optional

import os
import gym
import pickle
import numpy as np

import collections
from typing import Tuple, Union

from tqdm import tqdm
import jax
import jax.numpy as jnp
from functools import partial



Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


class ReplayBufferMulti():
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, n: int):

        self.observations = np.empty((capacity, n, *observation_space.shape),
                                dtype=observation_space.dtype)
        self.actions = np.empty((capacity, n, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, n), dtype=np.float32)
        self.masks = np.empty((capacity, n), dtype=np.float32)
        self.dones_float = np.empty((capacity, n), dtype=np.float32)
        self.next_observations = np.empty((capacity, n, *observation_space.shape),
                                     dtype=observation_space.dtype)
        self.n = n

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity
        
        # for saving the buffer
        self.n_parts = 4
        assert self.capacity % self.n_parts == 0

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=(batch_size))
        return Batch(observations=self.observations[indx].transpose((1, 0, 2)),
                     actions=self.actions[indx].transpose((1, 0, 2)),
                     rewards=self.rewards[indx].transpose((1, 0)),
                     masks=self.masks[indx].transpose((1, 0)),
                     next_observations=self.next_observations[indx].transpose((1, 0, 2)))

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):

        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def save(self, data_path: str):
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts
        
        for i in range(self.n_parts):
            data_chunk = [
                self.observations[i*chunk_size : (i+1)*chunk_size],
                self.actions[i*chunk_size : (i+1)*chunk_size],
                self.rewards[i*chunk_size : (i+1)*chunk_size],
                self.masks[i*chunk_size : (i+1)*chunk_size],
                self.dones_float[i*chunk_size : (i+1)*chunk_size],
                self.next_observations[i*chunk_size : (i+1)*chunk_size]
            ]
            
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))

    def load(self, data_path: str):
        chunk_size = self.capacity // self.n_parts
        total_size = 0
        
        for i in range(self.n_parts):            
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))
            total_size += len(data_chunk[0])

            self.observations[i*chunk_size : (i+1)*chunk_size], \
            self.actions[i*chunk_size : (i+1)*chunk_size], \
            self.rewards[i*chunk_size : (i+1)*chunk_size], \
            self.masks[i*chunk_size : (i+1)*chunk_size], \
            self.dones_float[i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[i*chunk_size : (i+1)*chunk_size] = data_chunk
            
        if self.capacity != total_size:
            print('WARNING: buffer capacity does not match size of loaded data!')
        self.insert_index = 0
        self.size = total_size
