from functools import partial
from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init
class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class D4PGEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (2, 1, 1, 1)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"
    kernel_init: Optional[Callable] = None
    dtype: Any = jnp.float32
    dropout_rate:float = 0.2
    @nn.compact
    def __call__(self, observations: jnp.ndarray,training=True) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)
        observations=observations.astype(jnp.float32)
        x = jnp.reshape(observations, (*observations.shape[:-2], -1))
        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            kernel_init= self.kernel_init or default_init
            x = nn.Conv(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                kernel_init=kernel_init(),
                padding=self.padding,
            )(x)
            # x = nn.LayerNorm()(x)
            x = nn.swish(x)

        return x.reshape((*x.shape[:-3], -1))
