import numpy as np
from flax.training.train_state import TrainState

from jaxrl2.agents.common import action_dist_jit, eval_actions_jit, eval_log_prob_jit, sample_actions_jit,extract_feature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import PRNGKey

from flax import struct
from flax.training.train_state import TrainState

class Agent(object):
    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
            self._actor.apply_fn, self._actor.params, observations
        )
        return np.asarray(actions)
    def eval_log_probs(self, batch: DatasetDict) -> float:
        return eval_log_prob_jit(self._actor.apply_fn, self._actor.params, batch)
    
    def action_dist(self, batch: DatasetDict):
        return action_dist_jit(self._actor.apply_fn, self._actor.params, batch)

    def extract_features(self, batch:DatasetDict):
        return np.asarray(extract_feature(self._actor.apply_fn, self._actor.params,batch))


    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng, self._actor.apply_fn, self._actor.params, observations
        )
        self._rng = rng
        return np.asarray(actions)
