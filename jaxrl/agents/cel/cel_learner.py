"""Implementations of RedQ.
https://arxiv.org/abs/2101.05982
"""

import functools
from typing import Optional, Sequence, Tuple, Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.cel.actor import update as update_actor
from jaxrl.agents.cel.critic import target_update
from jaxrl.agents.cel.critic import update as update_critic, ensemble_update
from jaxrl.agents.cel import temperature
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, Params, get_param_count


@functools.partial(jax.jit,
                   static_argnames=('n'))
@functools.partial(jax.vmap, in_axes=(0, None, None))
def _sample_index(rng: PRNGKey, n: int, probs: jnp.ndarray):
    # WARNING: probs should not change across calls. Otherwise compilation cost
    # will be expensive
    rng, key = jax.random.split(rng)
    assert probs.shape == (n,)
    index = jax.random.choice(key, a=n, p=probs)
    return rng, index

@functools.partial(jax.jit,
                   static_argnames=('mode'))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None, 0))
def _sample_actions(rng, actor, observations, mode: str, agent_id: int):
    NUM_SAMPLES = 10
    dist = actor(observations)
    rng, key = jax.random.split(rng)
    # (N, A)
    all_actions = dist.sample(seed=key)
    assert all_actions.ndim == 2
    if mode == 'avg':
        action = all_actions.mean(axis=0)
    elif mode == 'single':
        action = all_actions[agent_id]
    else:
        raise ValueError(f'Invalid mode {mode}')

    
    N, A = all_actions.shape
    if NUM_SAMPLES == 1:
        samples = all_actions
    else:
        # It is ok to reuse the key here, no one cares
        samples = dist.sample(NUM_SAMPLES, seed=key)
        samples = samples.mean(axis=0)
        assert samples.shape == (N, A)
    # Compute pairwise distance
    # (N,1)
    xx = (samples ** 2).sum(axis=1, keepdims=True)
    # (1,N)
    yy = (samples.T ** 2).sum(axis=0, keepdims=True)
    # (N, N)
    xy = samples @ samples.T
    # (N, 1) + (1, N) - 2 * (N, N). Note diagonal entries will be zero
    pairwise_dist = (xx + yy - 2 * xy).sum(axis=[0, 1]) / (2 * N * (N - 1))
    # Normalized by action dim
    pairwise_dist /= action.shape[-1]
    info = {
        'pairwise_dist': pairwise_dist
    }

    return rng, action, info

@functools.partial(jax.jit,
                   static_argnames=('backup_entropy', 'update_target',
                                    'update_policy', 'cel', 'cel_type',
                                    'aux_huber'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, update_target: bool, 
    update_policy: bool, cel: bool, cel_type: str, aux_huber: bool, huber_delta: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    # To avoid specific a lot of stuff in vmap
    @jax.vmap
    def _update(rng: PRNGKey, actor: Model, critic: Model, target_critic: Model, temp: Model, batch: Batch):
        rng, key = jax.random.split(rng)
        new_critic, critic_info = update_critic(key,
                                                actor,
                                                critic,
                                                target_critic,
                                                temp,
                                                batch,
                                                discount,
                                                backup_entropy=backup_entropy,
                                                cel=cel,
                                                cel_type=cel_type,
                                                aux_huber=aux_huber,
                                                huber_delta=huber_delta,
                                                )
        if update_target:
            new_target_critic = target_update(new_critic, target_critic, tau)
        else:
            new_target_critic = target_critic

        if update_policy:
            rng, key = jax.random.split(rng)
            new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, cel=cel)
            new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                                      target_entropy)
            actor_info.pop('entropy')
        else:
            new_actor, actor_info = actor, {}
            new_temp, alpha_info = temp, {}

        return rng, new_actor, new_critic, new_target_critic, new_temp, {
            **critic_info,
            **actor_info,
            **alpha_info
        }
    return _update(rng, actor, critic, target_critic, temp, batch)


class CELLearner(object):

    def __init__(
            self,
            seed: int,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            actor_lr: float = 3e-4,
            critic_lr: float = 3e-4,
            temp_lr: float = 3e-4,
            n: int = 10,  # Number of critics.
            policy_update_delay: int = 20,  # See the original implementation.
            hidden_dims: Sequence[int] = (256, 256),
            discount: float = 0.99,
            tau: float = 0.005,
            target_update_period: int = 1,
            target_entropy: Optional[float] = None,
            backup_entropy: bool = True,
            init_temperature: float = 1.0,
            # init_mean: Optional[np.ndarray] = None,
            # policy_final_fc_init_scale: float = 1.0,
            nsteps: int = 1,
            share_batch: bool = True,
            cel: bool = False,
            tandem: bool = False,
            active_prob: float = 0.5,
            cel_type: str = 'q',
            aux_huber: bool = False,
            huber_delta: float = 1.0,
            ln_critic: bool = False,
            ln_params_critic: bool = True,
                
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905

        Arguments:
            cel: whether to use CERL loss
            tandem: use the tandem setting. n must be 2
            active_prob: probability of sampling the active agent when using tandem
            cel_type: 'q' or 'pi'
                q: the default in the paper
                pi: the target value comes from the agent itself, but the action for
                    q(s, a) comes from other agents
            aux_huber: for the auxiliary loss we use huber
            huber_delta: as the name suggests
        """

        action_dim = actions.shape[-1]
        assert cel_type in ['q', 'pi']

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.n = n
        self.policy_update_delay = policy_update_delay

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.cel = cel

        self.share_batch = share_batch
        self.seed = seed
        self.tandem = tandem
        self.cel_type = cel_type
        self.aux_huber = aux_huber
        self.huber_delta = huber_delta
        self.ln_critic = ln_critic
        self.ln_params_critic = ln_params_critic
        if self.tandem:
            assert self.n == 2
            self.active_prob = active_prob
        else:
            self.active_prob = None
        # For creating new agents
        rng = jax.vmap(jax.random.PRNGKey)(np.array(seed))

        @jax.jit
        @jax.vmap
        def _create_new_agent(rng: PRNGKey):
            rng, actor_key, critic_key, ensemble_key, temp_key = jax.random.split(rng, 5)
            broadcast_actions = jnp.repeat(actions[None], repeats=n, axis=0)
            obs_example = observations if share_batch else jnp.repeat(observations[None], repeats=n, axis=0)
            actor_def = policies.EnsembleNormalTanhPolicy(
                    hidden_dims,
                    action_dim,
                    num_pis=n,
                    share_batch=share_batch
            )
            actor = Model.create(actor_def,
                                 inputs=[actor_key, obs_example],
                                 tx=optax.adam(learning_rate=actor_lr))

            critic_def = critic_net.EnsembleDoubleCritic(hidden_dims, num_qs=n, share_batch=share_batch, cel=cel,
                                                         ln=ln_critic, ln_params=ln_params_critic)
            critic = Model.create(critic_def,
                                  inputs=[critic_key, obs_example, broadcast_actions],
                                  tx=optax.adam(learning_rate=critic_lr))


            target_critic = Model.create(
                critic_def, inputs=[critic_key, obs_example, broadcast_actions])

            temp = Model.create(temperature.EnsembleTemperature(init_temperature, num=n),
                                inputs=[temp_key],
                                tx=optax.adam(learning_rate=temp_lr))
            train_agent_id = 0
            eval_agent_id = 0
            return rng, actor, critic, target_critic, temp, train_agent_id, eval_agent_id

        self.create_new_agent = _create_new_agent
        self.rng, self.actor, self.critic, self.target_critic, self.temp, self.train_agent_id, self.eval_agent_id = self.create_new_agent(rng)
        print('Number of parameters: {}'.format(get_param_count(self.actor.params) + get_param_count(self.critic.params) + get_param_count(self.temp)))

        self.step = 0
        self.eval_policy = None

    def set_eval_policy(self, eval_policy: str, eval_agent_id: Optional[int] = None):
        assert eval_policy in ['avg', 'random', 'single']
        self.eval_policy = eval_policy
        if eval_policy == 'single':
            assert isinstance(eval_agent_id, int)
            self.eval_agent_id = jnp.full(shape=len(self.seed), fill_value=eval_agent_id)
        else:
            assert eval_agent_id is None

    def begin_episode(self, mask: np.ndarray, eval_mode: bool):
        if eval_mode:
            if self.eval_policy == 'random':
                probs = jnp.ones(shape=self.n) / self.n
                self.rng, new_eval_agent_id = _sample_index(self.rng, self.n, probs)
                self.eval_agent_id = jnp.where(mask, new_eval_agent_id, self.eval_agent_id)
            elif self.eval_policy == 'avg':
                # Would not be used anyways
                pass
            elif self.eval_policy == 'single':
                # Set when set_eval_policy
                pass
            else:
                raise ValueError(f'Unknown eval policy {self.eval_policy}')
        else:
            if self.tandem:
                probs = jnp.array([self.active_prob, 1.0 - self.active_prob])
            else:
                probs = jnp.ones(shape=self.n) / self.n
            self.rng, new_train_agent_id = _sample_index(self.rng, self.n, probs)
            self.train_agent_id = jnp.where(mask, new_train_agent_id, self.train_agent_id)


    def reset(self):
        self.rng, self.actor, self.critic, self.target_critic, self.temp = self.create_new_agent(self.rng)


    def sample_actions(self,
                       observations: np.ndarray,
                       eval_mode: bool
                       ) -> jnp.ndarray:
        if eval_mode:
            agent_id = self.eval_agent_id
            if self.eval_policy == 'random' or self.eval_policy == 'single':
                mode = 'single'
            elif self.eval_policy == 'avg':
                mode = 'avg'
            else:
                raise ValueError(f'Invalid eval_policy {self.eval_policy}')
        else:
            agent_id = self.train_agent_id
            mode = 'single'


        self.rng, actions, info = _sample_actions(self.rng, self.actor,
                                                  observations, mode, agent_id)
        actions = np.asarray(actions)
        info = jax.tree_map(lambda x: np.asarray(x), info)

        return np.clip(actions, -1, 1), info

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            update_target=self.step % self.target_update_period == 0,
            update_policy=self.step % self.policy_update_delay == 0,
            cel=self.cel,
            cel_type=self.cel_type,
            aux_huber=self.aux_huber,
            huber_delta=self.huber_delta,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

