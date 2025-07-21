from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import flax
import flax.linen as nn
from flax import struct
import gymnasium as gym
import jax.numpy as jnp
import numpy as np

# Common Type Definitions
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

@struct.dataclass
class Batch:
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    masks: jnp.ndarray
    next_observations: jnp.ndarray
    discounts: jnp.ndarray

# --- Replay Buffer ---

class ReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            mask: float, next_obs: np.ndarray, discount: float):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.masks[self.ptr] = mask
        self.next_observations[self.ptr] = next_obs
        self.discounts[self.ptr] = discount
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        # Single transfer operation
        batch_data = {
            'observations': self.observations[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'masks': self.masks[idx],
            'next_observations': self.next_observations[idx],
            'discounts': self.discounts[idx]
        }
        # Transfer all data at once to reduce overhead
        batch_data_jax = jax.device_put(batch_data)
        return Batch(
            observations=batch_data_jax['observations'],
            actions=batch_data_jax['actions'],
            rewards=batch_data_jax['rewards'],
            masks=batch_data_jax['masks'],
            next_observations=batch_data_jax['next_observations'],
            discounts=batch_data_jax['discounts']
        )

    def __len__(self) -> int:
        return self.size

# --- Common Network Utilities ---

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False # Changed default to False for typical MLP usage
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x 
    

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax.nn import initializers

PRNGKey = Any
Array = Any
Shape = tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


class BatchRenorm(Module):
    """BatchRenorm Module (https://arxiv.org/abs/1702.03275).
    Adapted from flax.linen.normalization.BatchNorm

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer.

    Usage Note:
    If we define a model with BatchRenorm, for example::

      BRN = BatchRenorm(use_running_average=False, momentum=0.99, epsilon=0.001, dtype=jnp.float32)

    The initialized variables dict will contain in addition to a 'params'
    collection a separate 'batch_stats' collection that will contain all the
    running statistics for all the BatchRenorm layers in a model::

      vars_initialized = BRN.init(key, x)  # {'params': ..., 'batch_stats': ...}

    We then update the batch_stats during training by specifying that the
    `batch_stats` collection is mutable in the `apply` method for our module.::

      vars_in = {'params': params, 'batch_stats': old_batch_stats}
      y, mutated_vars = BRN.apply(vars_in, x, mutable=['batch_stats'])
      new_batch_stats = mutated_vars['batch_stats']

    During eval we would define BRN with `use_running_average=True` and use the
    batch_stats collection from training to set the statistics.  In this case
    we are not mutating the batch statistics collection, and needn't mark it
    mutable::

      vars_in = {'params': params, 'batch_stats': training_batch_stats}
      y = BRN.apply(vars_in, x)

    Attributes:
      use_running_average: if True, the statistics stored in batch_stats will be
        used. Else the running statistics will be first updated and then used to normalize.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
        examples on the first two and last two devices. See `jax.lax.psum` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    warmup_steps: int = 100_000
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    # This parameter was added in flax.linen 0.7.2 (08/2023)
    # commented out to be compatible with a wider range of jax versions
    # TODO: re-activate in some months (04/2024)
    # use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Normalizes the input using batch statistics.

        NOTE:
        During initialization (when `self.is_initializing()` is `True`) the running
        average of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = merge_param("use_running_average", self.use_running_average, use_running_average)
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable("batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape)

        r_max = self.variable(
            "batch_stats",
            "r_max",
            lambda s: s,
            3,
        )
        d_max = self.variable(
            "batch_stats",
            "d_max",
            lambda s: s,
            5,
        )
        steps = self.variable(
            "batch_stats",
            "steps",
            lambda s: s,
            0,
        )

        if use_running_average:
            custom_mean = ra_mean.value
            custom_var = ra_var.value
        else:
            batch_mean, batch_var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                # use_fast_variance=self.use_fast_variance,
            )
            if self.is_initializing():
                custom_mean = batch_mean
                custom_var = batch_var
            else:
                std = jnp.sqrt(batch_var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                # scale
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                # bias
                d = jax.lax.stop_gradient((batch_mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)

                # BatchNorm normalization, using minibatch stats and running average stats
                # Because we use _normalize, this is equivalent to
                # ((x - x_mean) / sigma) * r + d = ((x - x_mean) * r + d * sigma) / sigma
                # where sigma = sqrt(var)
                affine_mean = batch_mean - d * jnp.sqrt(batch_var) / r
                affine_var = batch_var / (r**2)

                # Note: in the original paper, after some warmup phase (batch norm phase of 5k steps)
                # the constraints are linearly relaxed to r_max/d_max over 40k steps
                # Here we only have a warmup phase
                is_warmed_up = jnp.greater_equal(steps.value, self.warmup_steps).astype(jnp.float32)
                custom_mean = is_warmed_up * affine_mean + (1.0 - is_warmed_up) * batch_mean
                custom_var = is_warmed_up * affine_var + (1.0 - is_warmed_up) * batch_var

                ra_mean.value = self.momentum * ra_mean.value + (1.0 - self.momentum) * batch_mean
                ra_var.value = self.momentum * ra_var.value + (1.0 - self.momentum) * batch_var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


# Adapted from simba: https://github.com/SonyResearch/simba
class SimbaResidualBlock(nn.Module):
    hidden_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    # "the MLP is structured with an inverted bottleneck, where the hidden
    # dimension is expanded to 4 *  hidden_dim"
    scale_factor: int = 4
    norm_layer: type[nn.Module] = nn.LayerNorm

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.norm_layer()(x)
        x = nn.Dense(self.hidden_dim * self.scale_factor, kernel_init=nn.initializers.he_normal())(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        return residual + x


    


        
@jax.jit
def sample_actions(state: struct.dataclass, observations: jnp.ndarray, 
                   deterministic: bool = False) -> Tuple[struct.dataclass, jnp.ndarray]:
    """Sample actions from the policy.
    If deterministic is True, return the tanh of mean of base distribution instead of sampling."""
    rng, key = jax.random.split(state.rng)
    
    
    dist = state.actor.apply_fn({'params': state.actor.params}, observations)
    base_mean = dist.distribution.mean() 
    
    # Get actions based on whether we want deterministic or stochastic behavior
    actions = jax.lax.cond(
        deterministic,
        lambda: jnp.tanh(base_mean),  # Apply tanh to the mean of base distribution
        lambda: dist.sample(seed=key)  # Sample from transformed distribution
    )
    
    new_state = state.replace(rng=rng)  # Update RNG in the new state
    return new_state, actions



# --- Evaluation Function ---
def evaluate(agent: struct.dataclass, env: gym.Env, num_episodes: int,discount: float) -> Tuple[struct.dataclass, float]:
    """Evaluates the agent's performance in the environment."""
    total_reward = 0.0
    total_disc_reward = 0.0
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        disc_episode_reward = 0.0
        gamma = 1.0
        while not done:
            # Use agent's sample_eval method for deterministic evaluation actions
            agent, action = agent.sample_eval(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            disc_episode_reward += gamma * reward
            gamma *= discount
            
        total_reward += episode_reward
        total_disc_reward += disc_episode_reward
    return agent, total_reward / num_episodes, total_disc_reward / num_episodes
