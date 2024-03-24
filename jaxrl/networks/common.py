import os
import pickle
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import math

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

def iter_norm(iterator):
    return jnp.sqrt(sum((x**2).sum() for x in iterator))

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

Array = Any

def get_param_count(params):
    return sum(jnp.prod(jnp.array(x.shape)) for x in jax.tree_util.tree_leaves(params))

def hyperbolic_schedule(progress: float, gamma_max: float, gamma_min: float = 0.0):


    horizon_max = 1.0 / (1.0 - gamma_max)
    horizon_min = 1.0 / (1.0 - gamma_min)
    current_horizon = (horizon_max - horizon_min) * progress + horizon_min

    return 1 - 1.0 / current_horizon
def exponential_schedule(progress: float, gamma_max: float, gamma_min: float = 0.0):


    horizon_max = 1.0 / (1.0 - gamma_max)
    horizon_min = 1.0 / (1.0 - gamma_min)
    current_horizon = math.exp(math.log(horizon_min) + progress * math.log(horizon_max / horizon_min))

    return 1 - 1.0 / current_horizon

def discount_schedule(progress: float, gamma_max: float, gamma_min: float = 0.0, max_ratio: float = 0.5):

    anneal_ratio = (1.0 - max_ratio) / 2.0
    if progress <= anneal_ratio:
        progress = 0.5 * progress / anneal_ratio 
    elif progress > 1.0 - anneal_ratio:
        progress = 0.5 + 0.5 * (progress - (1.0 - anneal_ratio)) / anneal_ratio
    else:
        progress = 0.5  # Use gamma max

    if progress <= 0.5:
        ratio = 2 * progress
    else:
        assert 0.5 < progress <= 1.0
        ratio = 2 * (1.0 - progress)
    # anneal_ratio = (1.0 - max_ratio)
    # if progress <= anneal_ratio:
        # progress = progress / anneal_ratio 
    # else:
        # progress = 1.0

    # ratio = progress
        

    horizon_max = 1.0 / (1.0 - gamma_max)
    horizon_min = 1.0 / (1.0 - gamma_min)
    current_horizon = ratio * (horizon_max - horizon_min) + horizon_min 
    return 1 - 1.0 / current_horizon

def symmetric_hyperbolic_cosine_schedule(progress: float, gamma_max: float, gamma_min: float = 0.0, max_ratio: float = 0.5):

    anneal_ratio = (1.0 - max_ratio) / 2.0
    if progress <= anneal_ratio:
        progress = 0.5 * progress / anneal_ratio 
    elif progress > 1.0 - anneal_ratio:
        progress = 0.5 + 0.5 * (progress - (1.0 - anneal_ratio)) / anneal_ratio
    else:
        progress = 0.5  # Use gamma max

    ratio = abs(math.sin(progress * math.pi))

    horizon_max = 1.0 / (1.0 - gamma_max)
    horizon_min = 1.0 / (1.0 - gamma_min)
    current_horizon = ratio * (horizon_max - horizon_min) + horizon_min 
    return 1 - 1.0 / current_horizon

def symmetric_hyperbolic_schedule(progress: float, gamma_max: float):
    if progress <= 0.5:
        transformed_progress = progress * 2
    else:
        transformed_progress = (1.0 - progress) * 2

    horizon = 1.0 / (1.0 - gamma_max)
    return 1 - 1.0 / ((horizon - 1) * transformed_progress + 1)


class PNorm(nn.Module):

  epsilon: float = 1e-6
  use_scale: bool = True
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones

  @nn.compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + self.epsilon)
    if self.use_scale:
        x = x * self.param('scale', self.scale_init, ())
    return x

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    fn: bool = False
    ln: bool = False
    ln_params: bool = True
    pn: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if self.pn and i == len(self.hidden_dims) - 1:
                x = PNorm()(x)
            if self.fn and i == len(self.hidden_dims) - 1:
                x = x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-6)
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.ln:
                    if self.ln_params:
                        x = nn.LayerNorm()(x)
                    else:
                        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        grad_norm = tree_norm(grads)
        info['grad_norm'] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)

def save_model(path: str, model: Model):
    state_dict = flax.serialization.to_state_dict(model)
    with open(path, 'wb') as f:
        pickle.dump(state_dict, f)

def load_model(path: str, model: Model):
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    model = flax.serialization.from_state_dict(model, state_dict)
    return model


def split_tree(tree, key):
    tree_head = tree.unfreeze()
    tree_enc = tree_head.pop(key)
    tree_head = flax.core.FrozenDict(tree_head)
    tree_enc = flax.core.FrozenDict(tree_enc)
    return tree_enc, tree_head


