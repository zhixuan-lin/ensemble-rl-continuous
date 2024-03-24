from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import InfoDict, Model



class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

class EnsembleTemperature(nn.Module):
    initial_temperature: float = 1.0
    num: int = 2

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (self.num,), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(temp: Model, entropy: float,
           target_entropy: float) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        assert temperature.shape == entropy.shape
        assert temperature.ndim == entropy.ndim == 1
        temp_loss = (temperature * (entropy - target_entropy)).sum(axis=0)
        return temp_loss, {
            'temperature': temperature.mean(),
            'temperature_max': temperature.max(),
            'temperature_min': temperature.min(),
            'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')

    return new_temp, info
