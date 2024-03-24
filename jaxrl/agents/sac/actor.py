from typing import Tuple

import flax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch, l2: float, td3_cdq: bool) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        if td3_cdq:
            q = q1
        else:
            q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        w_norm_sq = jnp.sum(jnp.array(list(map(lambda x: (x**2).sum(),
                                               flax.traverse_util.ModelParamTraversal(
                                                   lambda k, v: 'kernel' in k).iterate(actor_params)))))
        actor_loss += l2 * w_norm_sq
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_params),
            'actor_action': jnp.mean(jnp.abs(actions))
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['actor_gnorm'] = info.pop('grad_norm')

    return new_actor, info
