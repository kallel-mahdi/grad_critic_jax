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

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            mask: float, next_obs: np.ndarray):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.masks[self.ptr] = mask
        self.next_observations[self.ptr] = next_obs

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
            'next_observations': self.next_observations[idx]
        }
        # Transfer all data at once to reduce overhead
        batch_data_jax = jax.device_put(batch_data)
        return Batch(
            observations=batch_data_jax['observations'],
            actions=batch_data_jax['actions'],
            rewards=batch_data_jax['rewards'],
            masks=batch_data_jax['masks'],
            next_observations=batch_data_jax['next_observations']
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
def evaluate(state: struct.dataclass, env: gym.Env, num_episodes: int) -> Tuple[struct.dataclass, float]:
    """Evaluates the agent's performance in the environment."""
    total_reward = 0.0
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Use deterministic actions for evaluation
            state, action = sample_actions(state, observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        total_reward += episode_reward
        
    return state, total_reward / num_episodes
