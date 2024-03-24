"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)

class CriticSep(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    fn: bool = False
    ln: bool = False
    pn: bool = False
    ln_params: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        obs_enc = MLP((self.hidden_dims[0],),
                      activate_final=True,
                      name='obs_net'
                     )(observations)
        action_enc = MLP((self.hidden_dims[0],),
                      activate_final=True,
                      name='action_net'
                     )(actions)
        inputs = jnp.concatenate([obs_enc, action_enc], -1)
        critic = MLP((*self.hidden_dims[1:], 1),
                     activations=self.activations,
                     fn=self.fn,
                     ln=self.ln,
                     ln_params=self.ln_params,
                     pn=self.pn)(inputs)
        return jnp.squeeze(critic, -1)

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    fn: bool = False
    ln: bool = False
    pn: bool = False
    ln_params: bool = True
    squeeze: bool = True
    output_dims: int = 1

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, self.output_dims),
                     activations=self.activations,
                     fn=self.fn,
                     ln=self.ln,
                     ln_params=self.ln_params,
                     pn=self.pn)(inputs)
        if self.squeeze:
            return jnp.squeeze(critic, -1)
        else:
            return critic


# class DoubleCritic(nn.Module):
    # hidden_dims: Sequence[int]
    # activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    # @nn.compact
    # def __call__(self, observations: jnp.ndarray,
                 # actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # critic1 = Critic(self.hidden_dims,
                         # activations=self.activations)(observations, actions)
        # critic2 = Critic(self.hidden_dims,
                         # activations=self.activations)(observations, actions)
        # return critic1, critic2
class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    fn: bool = False
    ln: bool = False
    ln_params: bool = True
    pn: bool = False
    sep: bool = False
    squeeze: bool = True
    output_dims: int = 1

    @nn.compact
    def __call__(self, states, actions):

        assert not self.sep
        VmapCritic = nn.vmap(Critic if not self.sep else CriticSep,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        fn=self.fn,
                        ln=self.ln,
                        pn=self.pn,
                        ln_params=self.ln_params,
                        squeeze=self.squeeze,
                        output_dims=self.output_dims)(states, actions)
        return qs

class EnsembleDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    share_batch: bool = True
    cel: bool = False
    ln: bool = False
    ln_params: bool = True

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(DoubleCritic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=(None, 0) if self.share_batch else (0, 0),
                             out_axes=1,
                             axis_size=self.num_qs)
        if self.cel:
            squeeze = False
            output_dims = self.num_qs
        else:
            squeeze = True
            output_dims = 1
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        num_qs=2,
                        squeeze=squeeze,
                        output_dims=output_dims,
                        ln=self.ln,
                        ln_params=self.ln_params)(states, actions)
        return qs
