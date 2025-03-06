"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
from jaxrl2.networks.encoders.pretrained_resnet import PretrainedResNet
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.utils.misc import augment_batch
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from  flax.training import train_state

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.bc.actor_updater import log_prob_update
from jaxrl2.utils.augmentations import batched_random_crop, batched_random_cutout
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder,PlaceholderEncoder
from jaxrl2.networks.normal_policy import UnitStdNormalPolicy, VariableStdNormalPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.types import Params, PRNGKey
import flaxmodels as fm

class TrainState(train_state.TrainState):
  batch_stats: Any

@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch_stats:Any,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats},observations)
    return dist.mode()

@jax.jit
def _update_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    rng, key = jax.random.split(rng)
    # if augument:
    rng, batch = augment_batch(key, batch,batched_random_crop)
    # rng, key = jax.random.split(rng)
    # rng, batch = augment_batch(key, batch,batched_random_cutout)

    # rng, new_actor, actor_info = log_prob_update(rng, actor, batch)
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params,batch_stats:Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist,updates = actor.apply_fn(
            {"params": actor_params,"batch_stats":batch_stats},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
            mutable=['batch_stats']
        )
        log_probs = dist.log_prob(batch["actions"])
        actor_loss = -log_probs.mean()
        return actor_loss,({"bc_loss": actor_loss},updates)

    grads, (info,updates) = jax.grad(loss_fn, has_aux=True)(actor.params,actor.batch_stats)
    new_actor = actor.apply_gradients(grads=grads)
    new_actor = new_actor.replace(batch_stats=updates['batch_stats'])
    return rng, new_actor, info



class PixelResNetBCLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        dropout_rate: Optional[float] = None,
        encoder: str = "d4pg",
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)


        encoder_def = partial(PretrainedResNet)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = VariableStdNormalPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate
        )
        actor_def = PixelMultiplexer(
            encoder=encoder_def, network=policy_def, latent_dim=latent_dim
        )
        params = actor_def.init(actor_key, observations)
        actor_params=params["params"]
        batch_stats=params["batch_stats"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            batch_stats=batch_stats,
            tx=optax.adam(learning_rate=actor_lr),
        )

        self._rng = rng
        self._actor = actor

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, info = _update_jit(self._rng, self._actor, batch)

        self._rng = new_rng
        self._actor = new_actor

        return info
    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
                    self._actor.apply_fn, self._actor.params,self._actor.batch_stats,observations
                )
        return np.asarray(actions)
