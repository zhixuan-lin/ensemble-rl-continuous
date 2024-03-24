from typing import Tuple, Callable

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm
import flax
import optax



# TODO: Not used, but maybe needed in the future
def ensemble_update(rng: PRNGKey, critic: Model, target_critic: Model,
                    actor: Model, temp: Model, create_new_agent: Callable,
                    index: int, reset_last_only: bool) -> Tuple[Model, Model]:
    new_critic_params, new_actor_params, new_temp_params = create_new_agent(rng)



    def update_model(model: Model, params: Params, name: str):
        assert name in ['critic', 'actor', 'critic_target', 'temp']
        def update_func_general(old, new):
            return jax.tree_map(lambda o, n: o.at[index].set(n.squeeze(axis=0)), old, new)

        def update_func(old, new):
            updated = old.unfreeze()
            if reset_last_only:
                if name == 'critic' or name == 'critic_target':
                    updated['VmapDoubleCritic_0']['VmapCritic_0']['MLP_0']['Dense_2'] = update_func_general(
                        old['VmapDoubleCritic_0']['VmapCritic_0']['MLP_0']['Dense_2'], new['VmapDoubleCritic_0']['VmapCritic_0']['MLP_0']['Dense_2']
                    )
                elif name == 'actor':
                    for i in [0, 1]:
                        updated['VmapNormalTanhPolicyParams_0'][f'Dense_{i}'] = update_func_general(
                            old['VmapNormalTanhPolicyParams_0'][f'Dense_{i}'], new['VmapNormalTanhPolicyParams_0'][f'Dense_{i}']
                        )
                else:
                    updated = update_func_general(old, new)
                return flax.core.freeze(updated)
            else:
                return update_func_general(old, new)

        if model.tx is not None:
            new_opt_state = model.tx.init(params['params'])
            updated_counts = update_func(model.opt_state[0].counts, new_opt_state[0].counts)
            updated_mu = update_func(model.opt_state[0].mu, new_opt_state[0].mu)
            updated_nu = update_func(model.opt_state[0].nu, new_opt_state[0].nu)
            updated_opt_state = (
                [model.opt_state[0]._replace(mu=updated_mu, nu=updated_nu, counts=updated_counts)] +
                model.opt_state[1:]
            )
        else:
            updated_opt_state = None

        updated_params = update_func(model.params, params['params'])
        return model.replace(params=updated_params, opt_state=updated_opt_state)

    updated_critic = update_model(critic, new_critic_params, 'critic')
    updated_target_critic = update_model(target_critic, new_critic_params, 'critic_target')
    updated_actor = update_model(actor, new_actor_params, 'actor')
    updated_temp = update_model(temp, new_temp_params, 'temp')

    return updated_critic, updated_target_critic, updated_actor, updated_temp


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)

def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, backup_entropy: bool,
           cel: bool, cel_type: str, aux_huber: bool, huber_delta: float,
           ) -> Tuple[Model, InfoDict]:

    B = batch.rewards.shape[0]
    N = critic.apply_fn.num_qs
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    # (B, *), (N, B, A)
    if cel and cel_type == 'pi':
        def get_next_q_for_one_member(next_action):
            # Just one member next_action (B, A) -> (N, B, A)
            next_action = jnp.repeat(next_action[None, :, :], N, axis=0)
            # (2, N, B, N)
            next_q = target_critic(batch.next_observations, next_action)
            return next_q
        # (2, N, B, N, N)
        next_qs = jax.vmap(get_next_q_for_one_member, in_axes=0, out_axes=-1)(next_actions)
        # (2, N, B, N)
        next_qs = jnp.diagonal(next_qs, axis1=3, axis2=4)
    else:
        next_qs = target_critic(batch.next_observations, next_actions)
    if cel:
        # next_qs (2, N, B, N), first N ensemble second N head
        assert next_qs.shape == (2, N, B, N)
        if cel_type == 'q':
            # (2, B, N)
            next_qs = jnp.diagonal(next_qs, axis1=1, axis2=3)
            # (2, N, B)
            next_qs = next_qs.transpose(0, 2, 1)
            assert next_qs.shape == (2, N, B)
        elif cel_type == 'pi':
            # DO nothing
            # (2, N, B, N) -> (2, N, N, B)
            next_qs = next_qs.transpose(0, 1, 3, 2)
            assert next_qs.shape == (2, N, N, B)
        else:
            raise ValueError()



    # (2, N, B) -> (N, B)
    # or (2, N, N, B) -> (N, N, B)
    # if cel and cel_type == 'pi':
    #     indices = jnp.arange(N)
    #     # (N, N, B)
    #     next_q = next_qs.mean(axis=0)
    #     # Min for diagonal, mean for other
    #     # No jnp.diagonal always put the axis to the right
    #     next_q = next_q.at[indices, indices, :].set(jnp.diagonal(next_qs, axis1=1, axis2=2).min(axis=0).T)
    # else:
    next_q = next_qs.min(axis=0)

    target_q = batch.rewards + discount * batch.masks * next_q
    # assert target_q.shape == next_log_probs.shape
    assert next_log_probs.shape == (N, B)
    assert target_q.shape == (N, B) or target_q.shape == (N, N, B)

    if backup_entropy:
        # temperature (N, 1)
        target_q -= discount * batch.masks * temp()[:, None] * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        if batch.actions.ndim < next_actions.ndim:
            # sharing batch. Doing a broadcast
            qs = critic.apply({'params': critic_params}, batch.observations,
                                 batch.actions[None].repeat(repeats=next_actions.shape[0], axis=0))
        else:
            assert False
            qs = critic.apply({'params': critic_params}, batch.observations,
                                 batch.actions)

        if cel:
            # qs (2, N1, B, N2) N1 ensemble, N2 head, target_q: (N, B)
            # Some weird but corret stuff
            assert qs.shape == (2, N, B, N)
            # (2, N1, N2, B)
            qs = qs.transpose(0, 1, 3, 2)
            assert qs.shape == (2, N, N, B)

            if cel_type == 'q':
                assert target_q.shape == (N, B), target_q.shape
                # (1, 1, N, B). All ensemble members share the same set of targets target
                target_q_reshape = target_q[None, None, :, :]
            elif cel_type == 'pi':
                # Do nothing
                target_q_reshape = target_q
                assert target_q_reshape.shape == (N, N, B)
            else:
                raise ValueError()
            if aux_huber:
                # (2, N, N, B) and (N, N, B)
                critic_loss_raw = 2.0 * optax.huber_loss(qs, target_q_reshape, delta=huber_delta)
                # (2, N, N, B) -> (2, B, N) -> (2, N, B)
                qs_main = jnp.diagonal(qs, axis1=1, axis2=2).transpose(0, 2, 1)
                # (1, 1, N, B) -> (1, N, B)
                target_main = target_q_reshape.squeeze(axis=0)
                indices = jnp.arange(N)
                # (2, N, N, B)
                critic_loss_raw = critic_loss_raw.at[:, indices, indices, :].set((qs_main - target_main) ** 2)
                # critic_loss_raw = ((qs - target_q_reshape) ** 2)
            else:
                # (2, N, N, B)
                critic_loss_raw = ((qs - target_q_reshape) ** 2)
            # Sum over (2, N, N), mean over B
            critic_loss = critic_loss_raw.mean(axis=3).sum(axis=[0, 1, 2])
            info = {
                'critic_loss': jnp.diagonal(critic_loss_raw, axis1=1, axis2=2).mean(),
                'q_min': jnp.mean(jnp.diagonal(qs, axis1=1, axis2=2).min(axis=-1)),
                'q_max': jnp.mean(jnp.diagonal(qs, axis1=1, axis2=2).max(axis=-1)),
                'q_mean': jnp.mean(jnp.diagonal(qs, axis1=1, axis2=2)),
                'q_all_mean': jnp.mean(qs),
                'q_all_min': jnp.min(qs, axis=(1, 2)).mean(),
                'q_all_max': jnp.max(qs, axis=(1, 2)).mean(),
                'r': batch.rewards.mean(),
                'entropy_reward': -(temp()[:, None] * next_log_probs).mean(),
                'critic_pnorm': tree_norm(critic_params),
            }
        else:
            # (2, N, B), sum over 2 and N, mean over batch
            critic_loss = ((qs - target_q) ** 2).mean(axis=2).sum(axis=[0, 1])
            info = {
                'critic_loss': critic_loss,
                'q_max': qs.max(axis=1).mean(),
                'q_min': qs.min(axis=1).mean(),
                'q_mean': jnp.mean(qs),
                'r': batch.rewards.mean(),
                'entropy_reward': -(temp()[:, None] * next_log_probs).mean(),
                'critic_pnorm': tree_norm(critic_params),
            }
        return critic_loss, info
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info['critic_gnorm'] = info.pop('grad_norm')

    return new_critic, info
