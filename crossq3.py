from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
from flax.training import train_state
from flax import struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
import distrax

# Import common components from utils
from utils import Batch, PRNGKey, Params, InfoDict, BatchRenorm

# Import base agent class
from base_agent import RLAgent, RLAgentState, RLAgentConfig

@struct.dataclass
class CrossQTD3Config(RLAgentConfig):
    actor_lr: float
    critic_lr: float
    hidden_dims: Sequence[int]
    discount: float
    policy_noise: float
    noise_clip: float
    policy_delay: int  # Frequency of delayed policy updates (same as TD3's policy_delay)
    exploration_noise: float
    max_action: float
    beta1:float
    final_fc_init_scale: float = 1.0
    # BatchRenorm specific params
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    use_batch_norm: bool = True
    n_critics: int = 2
    

# New state class for batch stats tracking
class BatchNormTrainState(train_state.TrainState):
    batch_stats: FrozenDict

@struct.dataclass
class CrossQTD3State(RLAgentState):
    rng: PRNGKey
    step: int
    actor: BatchNormTrainState
    critic: BatchNormTrainState
    config: CrossQTD3Config  # Store configuration in the state

# --- Deterministic Actor Network with BatchRenorm ---
class DeterministicActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float
    use_batch_norm: bool = True
    final_fc_init_scale: float = 1.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # Flatten input if needed - assuming x has shape [batch_size, *observation_dims]
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        # Apply BatchRenorm initially if used
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not train,)(x)
        else:
            x_dummy = BatchRenorm(use_running_average=not train)(x)
        
        # Standard MLP with BatchRenorm between layers
        for i, n_units in enumerate(self.hidden_dims):
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
            if self.use_batch_norm:
                x = BatchRenorm(use_running_average=not train,)(x)
            else:
                x_dummy = BatchRenorm(use_running_average=not train)(x)
        
        # Final layer for deterministic actions
        x = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.uniform(scale=self.final_fc_init_scale)
        )(x)
        
        # Apply tanh and scale to max_action
        return nn.tanh(x) * self.max_action

# --- Critic Network with BatchRenorm ---
class Critic(nn.Module):
    hidden_dims: Sequence[int]
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        
        # Flatten observations if needed
        if len(observations.shape) > 2:
            observations = observations.reshape(observations.shape[0], -1)
        
        x = jnp.concatenate([observations, actions], -1)

        if self.use_batch_norm:
            x = BatchRenorm(
                use_running_average=not train, )(x)
        else:
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        for n_units in self.hidden_dims:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)

            if self.use_batch_norm:
                x = BatchRenorm(use_running_average=not train, )(x)
            else:
                x_dummy = BatchRenorm(use_running_average=not train)(x)

        x = nn.Dense(1)(x)
        return x.squeeze(-1)

class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    use_batch_norm: bool = True
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions, train: bool = False):
        # Creates multiple Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(
            Critic,
            variable_axes={'params': 0, 'batch_stats': 0},  # Map over parameters and batch stats
            split_rngs={'params': True, 'batch_stats': True},  # Split RNGs for each critic
            in_axes=None,  # Inputs (states, actions) are shared
            out_axes=0,  # Stack outputs along the first axis
            axis_size=self.num_qs  # Number of critics to create
        )
        
        qs = VmapCritic(
            self.hidden_dims,
            use_batch_norm=self.use_batch_norm,
        )(states, actions, train)
        
        return qs

# --- Update Functions ---

def update_actor(key_actor: PRNGKey, state: CrossQTD3State, batch: Batch) -> Tuple[BatchNormTrainState, InfoDict]:

    def actor_loss_fn(actor_params):
        # Apply actor model to get deterministic actions
        variables = {'params': actor_params, 'batch_stats': state.actor.batch_stats}
        
        # Always use mutable=["batch_stats"] regardless of whether batch norm is used
        actions, new_model_state = state.actor.apply_fn(
            variables, batch.observations, train=True, mutable=['batch_stats']
        )

        # Evaluate actions using the critic network's apply_fn
        critic_variables = {'params': state.critic.params, 'batch_stats': state.critic.batch_stats}
        
        # Always use mutable=False for critic during actor update (no training)
        q_values = state.critic.apply_fn(
            critic_variables, batch.observations, actions, train=False, mutable=False
        )
        
        # Take first critic for actor loss (following TD3 convention)
        q = q_values[0]

        # Actor loss: maximize Q-value (so negate it)
        actor_loss = -q.mean()

        # Return loss, new batch_stats, and auxiliary info
        return actor_loss, (new_model_state, {'actor_loss': actor_loss})

    # Compute gradients and update actor model using TrainState's apply_gradients
    (actor_loss, (new_model_state, info)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)
    
    # Update actor
    new_actor = state.actor.apply_gradients(grads=grads)
    # Update batch_stats with new values - converting to FrozenDict
    new_actor = new_actor.replace(batch_stats=new_model_state['batch_stats'])

    return new_actor, info


def update_critic(key_critic: PRNGKey, state: CrossQTD3State, batch: Batch) -> Tuple[BatchNormTrainState, InfoDict]:
    # Target Policy Smoothing: Add noise to target actions
    noise = (jax.random.normal(key_critic, batch.actions.shape) * state.config.policy_noise
            ).clip(-state.config.noise_clip, state.config.noise_clip)
    
    # Get next actions from the actor for the *next* observations (no target network)
    variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
    next_actions = state.actor.apply_fn(variables, batch.next_observations, train=False, mutable=False)
    
    # Add clipped noise to next actions
    next_actions = (next_actions + noise).clip(-state.config.max_action, state.config.max_action)
    
    # The key innovation: process both current and next observation-action pairs through the critic
    # This ensures BatchRenorm statistics are calculated on the joint distribution
    joint_observations = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
    joint_actions = jnp.concatenate([batch.actions, next_actions], axis=0)
    
    def critic_loss_fn(critic_params):
        # Joint forward pass through critic with both current and next pairs
        variables = {'params': critic_params, 'batch_stats': state.critic.batch_stats}
        
        # Always use mutable=["batch_stats"] regardless of whether batch norm is used
        joint_q_values, new_model_state = state.critic.apply_fn(
            variables, joint_observations, joint_actions, train=True, mutable=['batch_stats']
        )
        
        # Split back into current and next Q-values
        batch_size = batch.observations.shape[0]
        current_q_values, next_q_values = joint_q_values[:, :batch_size], joint_q_values[:, batch_size:]
        
        # Compute target Q-values (minimum across critics, following TD3)
        min_next_q = jnp.min(next_q_values, axis=0)

        # Bellman target calculation (no entropy term like in SAC)
        target_q = batch.rewards + state.config.discount * batch.masks * min_next_q
        target_q = jax.lax.stop_gradient(target_q)
        
        # Compute losses for all critics
        critic_loss = ((current_q_values[0] - target_q)**2 + (current_q_values[1] - target_q)**2).mean()
        
        # Return loss, new batch_stats, and auxiliary info
        return critic_loss, (new_model_state, {
            'critic_loss': critic_loss,
            'q1': current_q_values[0].mean(),
            'q2': current_q_values[1].mean()
        })
    
    # Compute gradients and update critic
    (critic_loss, (new_model_state, info)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic.params)
    
    # Update critic
    new_critic = state.critic.apply_gradients(grads=grads)
    # Update batch_stats with new values - converting to FrozenDict
    new_critic = new_critic.replace(batch_stats=new_model_state['batch_stats'])
    
    return new_critic, info


# --- Update Steps (External JIT functions) ---

@jax.jit
def _crossq_td3_update_step(state: CrossQTD3State, batch: Batch) -> Tuple[CrossQTD3State, InfoDict]:
    """Single update step for all components (external JIT)."""
    rng, key_critic, key_actor = jax.random.split(state.rng, 3)

    # Update critic
    new_critic, critic_info = update_critic(key_critic, state, batch)

    # Update actor with policy delay
    should_update_actor = (state.step % state.config.policy_delay == 0)
    
    def _update_actor():
        # Update actor using updated critic
        temp_state = state.replace(critic=new_critic)
        new_actor, actor_info = update_actor(key_actor, temp_state, batch)
        return new_actor, actor_info
    
    def _no_update_actor():
        return state.actor, {'actor_loss': jnp.nan}
    
    new_actor, actor_info = jax.lax.cond(
        should_update_actor,
        lambda: _update_actor(),
        lambda: _no_update_actor()
    )

    # Create new state with updated values
    new_state = state.replace(
        actor=new_actor,
        critic=new_critic,
        rng=rng,
        step=state.step + 1
    )

    return new_state, {**critic_info, **actor_info}


@jax.jit
def _crossq_td3_sample_step(state: CrossQTD3State, observation: jnp.ndarray) -> Tuple[CrossQTD3State, jnp.ndarray]:
    """Samples an action using the policy with exploration noise and updates RNG state (external JIT)."""
    rng, key_noise = jax.random.split(state.rng)
    
    # Get deterministic action from policy
    variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
    action = state.actor.apply_fn(variables, observation, train=False, mutable=False)
    
    # Add exploration noise
    noise = jax.random.normal(key_noise, action.shape) * state.config.exploration_noise * state.config.max_action
    action = (action + noise).clip(-state.config.max_action, state.config.max_action)
    
    new_state = state.replace(rng=rng)
    return new_state, action

@jax.jit
def _crossq_td3_sample_eval_step(state: CrossQTD3State, observation: jnp.ndarray) -> jnp.ndarray:
    """Samples a deterministic action for evaluation (external JIT)."""
    variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
    action = state.actor.apply_fn(variables, observation, train=False, mutable=False)
    return action


# --- Factory Function ---
def create_crossq_td3_learner(
    seed: int,
    observations: jnp.ndarray,  # Sample observation for initialization
    actions: jnp.ndarray,       # Sample action for initialization
    config: CrossQTD3Config,
) -> CrossQTD3State:
    """Factory function to create CrossQ TD3 state."""
    
    # Initialize PRNG keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    
    # Initialize Actor network and state
    actor_def = DeterministicActor(
        [256,256],
        actions.shape[-1],
        max_action=config.max_action,
        final_fc_init_scale=config.final_fc_init_scale,
        use_batch_norm=config.use_batch_norm,
    )
    
    # Initialize actor
    actor_variables = actor_def.init(actor_key, observations, train=False)
    # Create BatchNormTrainState for actor
    actor = BatchNormTrainState.create(
        apply_fn=actor_def.apply,
        params=actor_variables['params'],
        batch_stats=actor_variables.get('batch_stats', FrozenDict({})),
        tx=optax.adam(learning_rate=config.actor_lr, b1=config.beta1)
    )
    
    # Initialize Critic network and state
    critic_def = DoubleCritic(
        config.hidden_dims,
        use_batch_norm=config.use_batch_norm,
        num_qs=config.n_critics
    )
    
    # Initialize critic
    critic_variables = critic_def.init(critic_key, observations, actions, train=False)
    
    # Create BatchNormTrainState for critic
    critic = BatchNormTrainState.create(
        apply_fn=critic_def.apply,
        params=critic_variables['params'],
        batch_stats=critic_variables.get('batch_stats', FrozenDict({})),
        tx=optax.adam(learning_rate=config.critic_lr, b1=config.beta1)
    )
    
    # Create the CrossQ TD3 state
    return CrossQTD3State(
        actor=actor,
        critic=critic,
        rng=rng,
        step=0,
        config=config
    )


# --- CrossQ TD3 Agent Class ---
@struct.dataclass
class CrossQTD3Agent(RLAgent):
    state: CrossQTD3State

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        agent_config: Dict,
        **kwargs  # Accept potential extra kwargs from base class
    ) -> "CrossQTD3Agent":
        """Creates the CrossQTD3Agent with its initial state."""
        
        # Create initial state
        initial_state = create_crossq_td3_learner(
            seed=seed,
            observations=observations,
            actions=actions,
            config=CrossQTD3Config(**agent_config),
        )
        
        return cls(state=initial_state)

    def update(self, batch: Batch) -> Tuple["CrossQTD3Agent", InfoDict]:
        """Performs a single CrossQ TD3 training step using the external JIT function."""
        new_state, info = _crossq_td3_update_step(self.state, batch)
        # Return a new agent instance wrapping the new state
        return CrossQTD3Agent(state=new_state), info

    def sample(self, observation: jnp.ndarray) -> Tuple["CrossQTD3Agent", jnp.ndarray]:
        """Samples an action stochastically using the external JIT function."""
        new_state, action = _crossq_td3_sample_step(self.state, observation)
        # Return a new agent instance wrapping the new state (with updated RNG)
        return CrossQTD3Agent(state=new_state), action

    def sample_eval(self, observation: jnp.ndarray) -> Tuple["CrossQTD3Agent", jnp.ndarray]:
        """Samples an action deterministically using the external JIT function."""
        action = _crossq_td3_sample_eval_step(self.state, observation)
        # Evaluation sampling doesn't change the agent's state (no RNG update)
        return self, action