import argparse
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
from flax.training import train_state
from flax import struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from jax.tree_util import tree_map
import distrax
import wandb  # Add wandb import




# Common Type Definitions
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

# Set Weights & Biases API key
import os
os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"



@struct.dataclass
class Batch:
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    masks: jnp.ndarray
    next_observations: jnp.ndarray

@struct.dataclass
class SACConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    gamma_lr: float = 3e-4  # Learning rate for gamma critic
    hidden_dims: Sequence[int] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    target_update_period: int = 1
    target_entropy: Optional[float] = None
    backup_entropy: bool = True
    init_temperature: float = 1.0
    gamma_correction: bool = False  # New flag for gamma correction
    num_gamma_params: int = 1  # Output size for gamma head

@struct.dataclass
class SACState:
    rng: PRNGKey
    step: int
    actor: train_state.TrainState
    critic: train_state.TrainState
    target_critic_params: Params  # Store target params directly
    gamma_critic: train_state.TrainState  # Add gamma critic
    target_gamma_critic_params: Params  # Add target gamma critic params
    temp: train_state.TrainState
    config: SACConfig  # Store configuration in the state




# --- Replay Buffer (remains unchanged) ---

class ReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box, capacity: int):
        self.capacity = capacity
        self.ptr = 0  # Pointer to the next insertion spot
        self.size = 0  # Current number of elements stored

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Pre-allocate arrays for efficiency
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros(capacity, dtype=np.float32)  # For discounting (1.0 if not done, 0.0 if done)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            mask: float, next_obs: np.ndarray):
        # Insert data at the current index
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.masks[self.ptr] = mask
        self.next_observations[self.ptr] = next_obs

        # Update insert index and buffer size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        # Sample indices uniformly from the stored data
        idx = np.random.randint(0, self.size, size=batch_size)
        return Batch(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            masks=self.masks[idx],
            next_observations=self.next_observations[idx]
        )

    def __len__(self) -> int:
        return self.size



# --- Common Network Utilities ---

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
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

# --- Policy Network ---


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False):
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
     
    
        log_stds = nn.Dense(self.action_dim,
                            kernel_init=default_init(
                                self.final_fc_init_scale))(outputs)


        log_stds = jnp.clip(log_stds, -10.0, 2.0)

        # TFP uses scale_diag for MultivariateNormalDiag
        base_dist = distrax.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds))

        if self.tanh_squash_distribution:
           
            return distrax.Transformed(
            base_dist, distrax.Block(distrax.Tanh(), ndims=1))
        else:
            # Returns the raw Normal distribution without tanh squashing.
            return base_dist
        
@jax.jit
def sample_actions(state: SACState, observations: jnp.ndarray, 
                   deterministic: bool = False) -> Tuple[SACState, jnp.ndarray]:
    """Sample actions from the policy.
    If deterministic is True, return the tanh of mean of base distribution instead of sampling."""
    rng, key = jax.random.split(state.rng)
    
    # Ensure observations have batch dimension
    observations = jnp.atleast_2d(observations)
    
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



# --- Critic Network ---

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)

class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):
        # Creates two Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0}, # Map over parameters for each critic
                             split_rngs={'params': True}, # Use different RNGs for parameter initialization
                             in_axes=None,               # Inputs (states, actions) are shared
                             out_axes=0,               # Stack outputs along the first axis
                             axis_size=self.num_qs)      # Number of critics to create
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations)(states, actions)
        return qs[0], qs[1] # Return the two Q-values separately

# --- Gamma Critic Network ---

class GammaCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_params: int = 1  # Output size for gamma parameters

    def setup(self):
        # Main network excluding final layer
        self.feature_net = MLP(self.hidden_dims, activations=self.activations, activate_final=True)
        # Gamma parameter output head
        self.gamma_head = nn.Dense(self.num_params, kernel_init=default_init())
    
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        features = self.feature_net(inputs)
        # Stop gradient flow from features
        features = jax.lax.stop_gradient(features)
        # Output gamma parameters
        gamma_params = self.gamma_head(features)
        return gamma_params

class DoubleGammaCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    num_params: int = None  # Output size for gamma parameters

    @nn.compact
    def __call__(self, states, actions):
        # Creates two GammaCritic networks and runs them in parallel
        VmapGammaCritic = nn.vmap(
            GammaCritic,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs
        )
        gamma_params = VmapGammaCritic(
            self.hidden_dims, 
            activations=self.activations,
            num_params=self.num_params
        )(states, actions)
        
        return gamma_params[0], gamma_params[1]  # Return the two sets of gamma parameters

# --- Temperature Parameter ---

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        # Parameter for the log of temperature, ensures temperature > 0.
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

def update_temperature(state: SACState, entropy: float) -> Tuple[train_state.TrainState, InfoDict]:
    # Defines the loss function for temperature optimization.
    def temperature_loss_fn(temp_params):
        temperature = state.temp.apply_fn({'params': temp_params})
        # Loss aims to match current entropy to target entropy.
        temp_loss = temperature * (entropy - state.config.target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    # Compute gradients and update temperature using TrainState's apply_gradients
    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(state.temp.params)
    new_temp = state.temp.apply_gradients(grads=grads)

    return new_temp, info


# --- SAC Actor Update ---

def update_actor(key_actor: PRNGKey, state: SACState, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    # Get temperature value (scalar)
    temperature = state.temp.apply_fn({'params': state.temp.params})

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Apply actor model to get action distribution using actor's apply_fn
        dist = state.actor.apply_fn({'params': actor_params}, batch.observations) # Pass training=True if needed

      
        actions,log_probs = dist.sample_and_log_prob(seed=key_actor)

        # Evaluate actions using the critic network's apply_fn
        q1, q2 = state.critic.apply_fn({'params': state.critic.params}, batch.observations, actions)
        q = jnp.minimum(q1, q2)

        # Actor loss: encourages actions with high Q-values and high entropy.
        actor_loss = (log_probs * temperature - q).mean()

        # Return loss and auxiliary info (entropy).
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    # Compute gradients and update actor model using TrainState's apply_gradients
    grads, info = jax.grad(actor_loss_fn, has_aux=True)(state.actor.params)
    
    # Define function for gamma correction
    def _apply_gamma_correction(grads_and_state):
        grads, state = grads_and_state
        
        # Get actions from current policy
        dist = state.actor.apply_fn({'params': state.actor.params}, batch.observations)
        policy_actions, _ = dist.sample_and_log_prob(seed=key_actor)
        
        # Get gamma values for policy actions
        gamma1_pi, gamma2_pi = state.gamma_critic.apply_fn(
            {'params': state.gamma_critic.params},
            batch.observations,
            policy_actions
        )
        
        # Get gamma values for batch actions
        gamma1_batch, gamma2_batch = state.gamma_critic.apply_fn(
            {'params': state.gamma_critic.params},
            batch.observations,
            batch.actions
        )
        
        # Compute average gamma values
        gamma_pi = (gamma1_pi + gamma2_pi) / 2.0
        gamma_batch = (gamma1_batch + gamma2_batch) / 2.0
        
        # Compute correction term and average over the batch dimension
        grad_correction = gamma_pi - gamma_batch
        mean_grad_correction = jnp.mean(grad_correction, axis=0)
        
        # Get the tree structure and shapes from the gradients
        tree_struct = jax.tree_util.tree_structure(grads)
        leaves = jax.tree_util.tree_leaves(grads)
        
        # Calculate cumulative sizes for splitting the correction vector
        cum_sizes = jnp.cumsum(jnp.array([0] + [x.size for x in leaves]))
        
        # Split the mean correction vector according to parameter sizes
        def split_correction(i, correction):
            start = cum_sizes[i]
            size = leaves[i].size
            # Use lax.dynamic_slice for JIT compatibility on the 1D mean correction
            sliced_correction = jax.lax.dynamic_slice(correction, [start], [size])
            return jnp.reshape(sliced_correction, leaves[i].shape)
        
        # Create a list of reshaped corrections
        reshaped_corrections = [split_correction(i, mean_grad_correction) for i in range(len(leaves))]
        
        # Reconstruct the correction pytree
        correction_pytree = jax.tree_util.tree_unflatten(tree_struct, reshaped_corrections)
        
        # Apply correction to gradients (shapes should now match)
        return jax.tree_util.tree_map(lambda g, c: g + c, grads, correction_pytree)

    # Define function for no correction
    def _no_gamma_correction(grads_and_state):
        grads, _ = grads_and_state
        return grads

    # Use jax.lax.cond to conditionally apply gamma correction
    grads = jax.lax.cond(
        state.config.gamma_correction,
        _apply_gamma_correction,
        _no_gamma_correction,
        (grads, state)
    )
    
    new_actor = state.actor.apply_gradients(grads=grads)

    return new_actor, info


# --- SAC Critic Update ---

def target_update(critic_params: Params, target_critic_params: Params, tau: float) -> Params:
    # Soft update of target critic parameters. Operates on params directly.
    new_target_params = tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic_params,
        target_critic_params)
    return new_target_params

def update_critic(key_critic: PRNGKey, state: SACState, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    # Get next actions and log probabilities from the actor for the *next* observations.
    dist = state.actor.apply_fn({'params': state.actor.params}, batch.next_observations) # Pass training=True if needed
  
    next_actions,next_log_probs = dist.sample_and_log_prob(seed=key_critic)

    # Compute target Q-values using the target critic parameters and critic's apply_fn.
    next_q1, next_q2 = state.critic.apply_fn({'params': state.target_critic_params}, batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    # Bellman target: reward + discounted future Q-value.
    target_q = batch.rewards + state.config.discount * batch.masks * next_q
    temperature = state.temp.apply_fn({'params': state.temp.params})
    target_q -= state.config.discount * batch.masks * temperature * next_log_probs

    # Stop gradient flow through the target
    target_q = jax.lax.stop_gradient(target_q)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Compute current Q-values using the main critic network.
        q1, q2 = state.critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)

        # Critic loss: Mean Squared Bellman Error (MSBE).
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        # Return loss and auxiliary info (Q-values).
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    # Compute gradients and update critic model using TrainState's apply_gradients
    grads, info = jax.grad(critic_loss_fn, has_aux=True)(state.critic.params)
    new_critic = state.critic.apply_gradients(grads=grads)

    return new_critic, info

def update_gamma_critic(key_gamma: PRNGKey, state: SACState, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    """Updates the gamma critic network using per-sample actor gradients."""
    
    # 1. Get next state actions from the policy
    dist = state.actor.apply_fn({'params': state.actor.params}, batch.next_observations)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key_gamma)
    
    # 2. Compute per-sample actor gradients for next observations
    temperature = state.temp.apply_fn({'params': state.temp.params})
    
    # Function to compute actor loss gradient for a single observation
    def actor_loss_fn(params, single_obs):
        # Ensure observation has batch dimension
        obs = jnp.expand_dims(single_obs, 0)
        
        # Get action distribution
        dist = state.actor.apply_fn({'params': params}, obs)
        actions, log_probs = dist.sample_and_log_prob(seed=key_gamma)
        
        # Get Q-values
        q1, q2 = state.critic.apply_fn({'params': state.critic.params}, obs, actions)
        min_q = jnp.minimum(q1, q2)
        
        # Actor loss
        return (temperature * log_probs - min_q).mean()
    
    # Vectorize gradient computation across batch
    batch_grad_fn = jax.vmap(
        lambda obs: jax.grad(actor_loss_fn)(state.actor.params, obs),
        in_axes=0
    )
    
    # Compute gradients for each observation
    per_sample_grads = batch_grad_fn(batch.next_observations)
    # Print shape of per_sample_grads
    
    
    # Flatten gradients to create concatenated representation
    flat_grads = []
    for leaf in jax.tree_util.tree_leaves(per_sample_grads):
        flat_grads.append(jnp.reshape(leaf, (leaf.shape[0], -1)))
    
    actor_next_grads = jnp.concatenate(flat_grads, axis=1)
    
    # 3. Get gamma values for next state from target gamma critic
    gamma1_next, gamma2_next = state.gamma_critic.apply_fn(
        {'params': state.target_gamma_critic_params},
        batch.next_observations,
        next_actions
    )
    
    # 4. Compute target: (1-done) * discount * (actor_grads + next_gamma)
    gamma_next_target = (gamma1_next + gamma2_next) / 2.0
    # jax.debug.print("actor_next_grads shape: {}", actor_next_grads.shape)
    # jax.debug.print("gamma_next_target shape: {}", gamma_next_target.shape)
    
    masks_expanded = jnp.expand_dims(batch.masks, axis=-1)  # Shape: (256, 1)
    next_gamma_value = state.config.discount * masks_expanded * (actor_next_grads + gamma_next_target)
    

    
    # Stop gradient through target
    next_gamma_value = jax.lax.stop_gradient(next_gamma_value)
    
    def gamma_loss_fn(gamma_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # 5. Get current gamma values
        gamma1, gamma2 = state.gamma_critic.apply_fn(
            {'params': gamma_params},
            batch.observations,
            batch.actions
        )
        
        # 6. Compute MSE loss
        gamma_loss = ((gamma1 - next_gamma_value)**2).mean() + ((gamma2 - next_gamma_value)**2).mean()
        
        return gamma_loss, {
            'gamma_loss': gamma_loss,
            'gamma1_mean': gamma1.mean(),
            'gamma2_mean': gamma2.mean()
        }
    
    # 7. Compute gradients and update gamma critic
    grads, info = jax.grad(gamma_loss_fn, has_aux=True)(state.gamma_critic.params)
    new_gamma_critic = state.gamma_critic.apply_gradients(grads=grads)
    
    return new_gamma_critic, info


# --- SAC Learner ---

@jax.jit
def _update_step(
    state: SACState,
    batch: Batch
) -> Tuple[SACState, InfoDict]:
    """Single update step for all components."""
    rng, key_critic, key_actor, key_gamma = jax.random.split(state.rng, 4)

    # Update critic
    new_critic, critic_info = update_critic(key_critic, state, batch)

    # Conditionally update target critic parameters
    def _update_target(state_and_new_critic):
        state, new_critic = state_and_new_critic
        return target_update(new_critic.params, state.target_critic_params, state.config.tau)

    def _no_update_target(state_and_new_critic):
        state, _ = state_and_new_critic
        return state.target_critic_params

    new_target_critic_params = jax.lax.cond(
        state.step % state.config.target_update_period == 0,
        _update_target,
        _no_update_target,
        (state, new_critic)
    )

    # Update gamma critic and target params only if gamma correction is enabled
    def _update_gamma(state):
        new_gamma_critic, gamma_info = update_gamma_critic(key_gamma, state, batch)
        new_target_gamma_params = target_update(
            new_gamma_critic.params,
            state.target_gamma_critic_params, 
            state.config.tau
        )
        return new_gamma_critic, new_target_gamma_params, gamma_info
    
    def _no_update_gamma(state):
        return state.gamma_critic, state.target_gamma_critic_params, {'gamma_loss': 0.0, 'gamma1_mean': 0.0, 'gamma2_mean': 0.0}
    
    new_gamma_critic, new_target_gamma_params, gamma_info = jax.lax.cond(
        state.config.gamma_correction,
        _update_gamma,
        _no_update_gamma,
        state
    )

    # Update actor (with gamma correction if enabled)
    new_actor, actor_info = update_actor(key_actor, 
                                        state.replace(
                                            critic=new_critic,
                                            gamma_critic=new_gamma_critic
                                        ), 
                                        batch)

    # Update temperature
    new_temp, alpha_info = update_temperature(state, actor_info['entropy'])

    # Create new state with updated values
    new_state = state.replace(
        actor=new_actor,
        critic=new_critic,
        target_critic_params=new_target_critic_params,
        gamma_critic=new_gamma_critic,
        target_gamma_critic_params=new_target_gamma_params,
        temp=new_temp,
        rng=rng,
        step=state.step + 1
    )

    return new_state, {**critic_info, **actor_info, **alpha_info, **gamma_info}


def create_sac_learner(
    seed: int,
    observations: jnp.ndarray,  # Sample observation for initialization
    actions: jnp.ndarray,       # Sample action for initialization
    config: Optional[SACConfig] = None,
    init_mean: Optional[jnp.ndarray] = None,
    policy_final_fc_init_scale: float = 1.0
) -> SACState:
    """Factory function to create SAC state with gamma critic."""
    if config is None:
        config = SACConfig()
    
    # Set default target entropy if not specified
    action_dim = actions.shape[-1]
    if config.target_entropy is None:
        config = config.replace(target_entropy=-action_dim / 2.0)
    
    # Initialize PRNG keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, gamma_critic_key, temp_key = jax.random.split(rng, 5)
    
    # Initialize Actor network and state
    actor_def = NormalTanhPolicy(
        config.hidden_dims,
        action_dim,
        init_mean=init_mean,
        final_fc_init_scale=policy_final_fc_init_scale)
    actor_params = actor_def.init(actor_key, observations)['params']
    actor = train_state.TrainState.create(
        apply_fn=actor_def.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=config.actor_lr)
    )
    
    # Initialize Critic network and state
    critic_def = DoubleCritic(config.hidden_dims)
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = train_state.TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=config.critic_lr)
    )
    
    # Initialize GammaCritic network and state
    # Compute total number of parameters in actor network
    actor_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_params))
    
    gamma_critic_def = DoubleGammaCritic(config.hidden_dims, num_params=actor_param_count)
    gamma_critic_params = gamma_critic_def.init(gamma_critic_key, observations, actions)['params']
    gamma_critic = train_state.TrainState.create(
        apply_fn=gamma_critic_def.apply,
        params=gamma_critic_params,
        tx=optax.adam(learning_rate=config.gamma_lr)
    )
    
    # Initialize Temperature parameter and state
    temp_def = Temperature(config.init_temperature)
    temp_params = temp_def.init(temp_key)['params']
    temp = train_state.TrainState.create(
        apply_fn=temp_def.apply,
        params=temp_params,
        tx=optax.adam(learning_rate=config.temp_lr)
    )
    
    # Create the SAC state with gamma critic
    return SACState(
        actor=actor,
        critic=critic,
        target_critic_params=critic_params,
        gamma_critic=gamma_critic,
        target_gamma_critic_params=gamma_critic_params,
        temp=temp,
        rng=rng,
        step=0,
        config=config
    )


# --- Evaluation Function ---
def evaluate(state: SACState, env: gym.Env, num_episodes: int) -> Tuple[SACState, float]:
    """Evaluates the agent's performance in the environment."""
    total_reward = 0.0
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Use deterministic actions for evaluation
            state, action = sample_actions(state, observation, deterministic=True)
            #action = np.asarray(action[0])  # First action from batch dimension
            action = action[0]
            
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        total_reward += episode_reward
        
    return state, total_reward / num_episodes


# --- Main Training Loop ---
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SAC JAX Implementation with Gradient Critic')
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='Walker2d-v4', help='Gym environment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Training parameters
    parser.add_argument('--max_steps', type=int, default=1_000_000, help='Maximum number of training steps')
    parser.add_argument('--start_steps', type=int, default=10_000, help='Steps before starting training')
    parser.add_argument('--eval_freq', type=int, default=10_000, help='Evaluate policy every N steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--replay_buffer_capacity', type=int, default=1_000_000, help='Replay buffer capacity')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    
    # Algorithm parameters
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--temp_lr', type=float, default=3e-4, help='Temperature learning rate')
    parser.add_argument('--gamma_lr', type=float, default=3e-4, help='Gamma critic learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256], help='Hidden layer dimensions')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--target_update_period', type=int, default=1, help='Target update period')
    parser.add_argument('--backup_entropy', action='store_true', default=True, help='Include entropy in backup')
    parser.add_argument('--init_temperature', type=float, default=1.0, help='Initial temperature value')
    parser.add_argument('--gamma_correction', action='store_true',default=False, help='Enable gamma correction for actor updates')
    parser.add_argument('--num_gamma_params', type=int, default=1, help='Number of gamma parameters to output')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',default=True, help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='sac_jax_gradient_critic', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (optional)')
    
    args = parser.parse_args()

    # Initialize Wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),  # Log hyperparameters
            sync_tensorboard=False,
            monitor_gym=True,  # Automatically log gym environments
            save_code=True,  # Save the main script
        )

    # Create environment
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)  # Separate env for evaluation

    # Set random seeds
    np.random.seed(args.seed)

    # Get observation and action space info
    obs_space = env.observation_space
    act_space = env.action_space
    sample_obs = jnp.zeros((1, obs_space.shape[0]), dtype=jnp.float32)
    sample_act = jnp.zeros((1, act_space.shape[0]), dtype=jnp.float32)

    # Create SAC config
    config = SACConfig(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        temp_lr=args.temp_lr,
        gamma_lr=args.gamma_lr,
        hidden_dims=tuple(args.hidden_dims),
        discount=args.discount,
        tau=args.tau,
        target_update_period=args.target_update_period,
        target_entropy=-act_space.shape[0] / 2.0,  # Default for continuous control
        backup_entropy=args.backup_entropy,
        init_temperature=args.init_temperature,
        gamma_correction=args.gamma_correction,
        num_gamma_params=args.num_gamma_params
    )

    # Initialize SAC agent
    state = create_sac_learner(
        seed=args.seed,
        observations=sample_obs,
        actions=sample_act,
        config=config
    )

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        observation_space=obs_space, 
        action_space=act_space, 
        capacity=args.replay_buffer_capacity
    )

    # Training loop
    observation, _ = env.reset(seed=args.seed)
    
    for step_num in tqdm(range(1, args.max_steps + 1)):
        if step_num < args.start_steps:
            # Sample random actions before training starts
            action = env.action_space.sample()
        else:
            # Sample actions from the policy
            state, action_batch = sample_actions(state, observation)
            action = action_batch[0]
        # Execute action in environment
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        mask = 1.0 - float(terminated)  # Mask is 1 if not terminated (for critic target)

        # Store transition in replay buffer
        replay_buffer.add(observation, action, reward, mask, next_observation)

        # Update observation
        observation = next_observation

        # Reset environment if episode ends
        if done:
            observation, _ = env.reset()

        # Perform agent update if enough steps have passed
        if step_num >= args.start_steps:
            batch = replay_buffer.sample(args.batch_size)
            state, update_info = _update_step(state, batch)

            # Log training metrics to wandb
            if args.use_wandb:
                wandb.log({'training': update_info}, step=step_num,commit=False)

        # Evaluate agent periodically
        if step_num % args.eval_freq == 0:
            eval_state, avg_reward = evaluate(state, eval_env, args.eval_episodes)
            # Update state with new RNG from evaluation
            state = state.replace(rng=eval_state.rng)
            print(f"---------------------------------------")
            print(f"Step: {step_num}, Evaluation Avg Reward: {avg_reward:.2f}")
            print(f"---------------------------------------")

            # Log evaluation metrics to wandb
            if args.use_wandb:
                wandb.log({'evaluation': {'avg_reward': avg_reward}}, step=step_num,commit=True)

    env.close()
    eval_env.close()

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

    print("Training finished.")

if __name__ == "__main__":
    main()

