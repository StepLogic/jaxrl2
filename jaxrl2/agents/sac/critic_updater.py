from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_critic(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    discount: float,
    backup_entropy: bool,
    critic_reduction: str,
) -> Tuple[TrainState, Dict[str, float]]:
    # breakpoint()
    dist = actor.apply_fn({"params": actor.params}, batch["next_observations"])
    rng, key = jax.random.split(key)
   
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn(
        {"params": target_critic.params}, batch["next_observations"], next_actions,training=True, rngs={'dropout': rng}
    )
    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()
    # value=jax.numpy.nan_to_num(            value=jax.numpy.nan_to_num(value))
    target_q = jax.numpy.nan_to_num(batch["rewards"]) + discount * batch["masks"] * next_q

    if backup_entropy:
        target_q -= (
            discount
            * batch["masks"]
            * temp.apply_fn({"params": temp.params})
            * next_log_probs
        )

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn(
            {"params": critic_params}, batch["observations"], batch["actions"],training=True,rngs={'dropout': rng}
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        # breakpoint()
        return critic_loss, {
            "critic_loss": critic_loss,
            "td_errors":jnp.mean(((qs - target_q) ** 2).swapaxes(1,0),axis=-1),
            "q": qs.mean(),
            "target_actor_entropy": -next_log_probs.mean(),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
