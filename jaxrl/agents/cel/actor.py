from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch, cel: bool) -> Tuple[Model, InfoDict]:

    B = batch.rewards.shape[0]
    N = critic.apply_fn.num_qs
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        qs = critic(batch.observations, actions)
        if cel:
            # qs (2, N, B, N)
            # (2, B, N)
            qs = jnp.diagonal(qs, axis1=1, axis2=3)
            # (2, N, B)
            qs = qs.transpose(0, 2, 1)
        assert qs.shape == (2, N, B)
        q = qs.min(axis=0)
        # (N, B)
        assert q.shape == log_probs.shape
        actor_loss = (log_probs * temp()[:, None] - q).mean(axis=1).sum(axis=0)
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(axis=1),  # (N,)
            'entropy_mean': -log_probs.mean(),
            'entropy_max': (-log_probs).mean(axis=-1).max(),
            'entropy_min': (-log_probs).mean(axis=-1).min(),
            'actor_pnorm': tree_norm(actor_params),
            'actor_action': jnp.mean(jnp.abs(actions))
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['actor_gnorm'] = info.pop('grad_norm')

    return new_actor, info
