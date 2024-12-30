import collections
import copy
from typing import Iterable, Optional

import gymnasium
import jax
import numpy as np
from flax.core import frozen_dict
from gymnasium.spaces import Box

from jaxrl2.data.dataset import DatasetDict, _sample
from jaxrl2.data.replay_buffer import ReplayBuffer
from collections import defaultdict

def relabel_obs_fn(data_dict: DatasetDict,max_len):
    # change goal to some achievable observation i.e next observation
    _virtual_goal=data_dict["next_observations"]["pixels"]
    _virtual_goal_heading=data_dict["next_observations"]["vector"][-2]
    data_dict["next_observations"]["vector"][-1]=_virtual_goal_heading
    data_dict["next_observations"]["goal"]=_virtual_goal
    data_dict["observations"]["goal"]=_virtual_goal
    data_dict["observations"]["vector"][-1]=_virtual_goal_heading    
    data_dict["dones"]=True
    data_dict["masks"]=0.0 
    data_dict["rewards"]=10.0 
    return data_dict          
class HindsightReplayBuffer(ReplayBuffer):
    def __init__(
        self, observation_space: gymnasium.Space, action_space: gymnasium.Space, capacity: int,relabel_obs_fn=None
    ):
        super().__init__(observation_space,action_space,capacity)
        self._episodes=defaultdict(lambda *:[])
        self._episode_idx=0
        self.n_sampled_goals=3
        self.relabel_obs_fn=relabel_obs_fn

    def insert(self, data_dict: DatasetDict):
        if data_dict["dones"]:
            self._episode_idx+=1
            # sample addtional goals
            _virtual_indices=random.choices(self._episodes[self._episode_idx],k=2)
            for  _v_idx in _virtual_indices:  
                _virtual_dict={}
                for k,v in self.dataset_dict.items():
                    _virtual_dict[k]=v[_v_idx]
                if callable(self.relabel_obs_fn)
                    _virtual_dict=self.relabel_obs_fn(_virtual_dict,max_len=max(self._episodes[self._episode_idx]))
                    super().insert(_virtual_dict)
            if (self._insert_index + 1) % self._capacity == 0
                self._episode_idx=0
                self._episodes[self._episode_idx]=[]
        super().insert(data_dict)
        self._episodes[self._episode_idx].append(self._insert_index)

        



