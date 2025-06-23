from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
from flax.training import train_state
from flax import struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_map
import distrax

# Import common components from utils
from utils import Batch , PRNGKey, Params, InfoDict
# Import base agent class
from base_agent import RLAgent, RLAgentState,RLAgentConfig # Added import
# Import networks
from networks import  DoubleCritic, StochasticActor, Temperature

@struct.dataclass
class SACConfig(RLAgentConfig):
    actor_lr: float
    critic_lr: float
    temp_lr: float
    hidden_dims: Sequence[int]
    discount: float
    tau: float
    target_update_period: int
    target_entropy: float
    backup_entropy: bool
    init_temperature: float
    policy_log_std_min: float 
    policy_log_std_max: float 
    policy_final_fc_init_scale: float 
    target_entropy_multiplier: float
    max_action: float
    policy_delay: int

@struct.dataclass
class SACState(RLAgentState): # Inherit from RLAgentState
    rng: PRNGKey
    step: int
    actor: train_state.TrainState
    critic: train_state.TrainState
    target_critic_params: Params  # Store target params directly
    temp: train_state.TrainState
    config: SACConfig  # Store configuration in the state





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



# --- SAC Update Steps (External JIT functions) ---

@jax.jit
def _sac_update_step(state: SACState, batch: Batch) -> Tuple[SACState, InfoDict]:
    """Single update step for all components (external JIT)."""
    rng, key_critic, key_actor = jax.random.split(state.rng, 3)

    # Update critic
    new_critic, critic_info = update_critic(key_critic, state, batch)
    state = state.replace(critic=new_critic)

    # Conditionally update target critic parameters
    def _update_target(state):
        return target_update(new_critic.params, state.target_critic_params, state.config.tau)

    def _no_update_target(state):
        return state.target_critic_params

    new_target_critic_params = jax.lax.cond(
        state.step % state.config.target_update_period == 0,
        _update_target,
        _no_update_target,
        state
    )

    # Conditional Actor and Temperature Updates (Delayed Policy Update)
    def _update_actor_and_temp(state):
        # Update actor using potentially updated critic
        new_actor, actor_info = update_actor(key_actor, state, batch)

        # Update temperature using potentially updated actor
        new_temp, alpha_info = update_temperature(state, actor_info['entropy'])
        
        return new_actor, new_temp, {**actor_info, **alpha_info}

    def _no_update_actor_and_temp(state):
        # Keep actor and temperature the same, return empty info
        return state.actor, state.temp, {'actor_loss': jnp.nan, 'entropy': jnp.nan, 'temperature': jnp.nan, 'temp_loss': jnp.nan}

    # Use jax.lax.cond for conditional execution on device
    new_actor, new_temp, policy_info = jax.lax.cond(
        state.step % state.config.policy_delay == 0,
        _update_actor_and_temp,
        _no_update_actor_and_temp,
        state
    )

    # Create new state with updated values
    new_state = state.replace(
        actor=new_actor,
        critic=new_critic,
        target_critic_params=new_target_critic_params,
        temp=new_temp,
        rng=rng,
        step=state.step + 1
    )

    return new_state, {**critic_info, **policy_info}


@jax.jit
def _sac_sample_step(state: SACState, observation: jnp.ndarray) -> Tuple[SACState, jnp.ndarray]:
    """Samples an action using the policy and updates RNG state (external JIT)."""
    rng, key = jax.random.split(state.rng)
    dist = state.actor.apply_fn({'params': state.actor.params}, observation)
    action = dist.sample(seed=key)
    new_state = state.replace(rng=rng)
    return new_state, action

@jax.jit
def _sac_sample_eval_step(state: SACState, observation: jnp.ndarray) -> jnp.ndarray:
    """Samples a deterministic action for evaluation (external JIT)."""
    dist = state.actor.apply_fn({'params': state.actor.params}, observation,temperature=.0)
    action = dist.sample(seed=state.rng)
    
    return action


# --- SAC Learner ---

# Removed the old _update_step function as it's replaced by _sac_update_step

# --- Factory Function ---
def create_sac_learner(
    seed: int,
    observations: jnp.ndarray,  # Sample observation for initialization
    actions: jnp.ndarray,       # Sample action for initialization
    config: Optional[SACConfig] = None,
) -> SACState:
    """Factory function to create SAC state."""
    
    # Set default target entropy if not specified
    action_dim = actions.shape[-1]
    
    # Initialize PRNG keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    
    # Initialize Actor network and state
    actor_def = StochasticActor(
        config.hidden_dims,
        action_dim,
        max_action=config.max_action,
        log_std_min=config.policy_log_std_min,
        log_std_max=config.policy_log_std_max,
        final_fc_init_scale=config.policy_final_fc_init_scale)
    
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
    
    # Initialize Temperature parameter and state
    temp_def = Temperature(config.init_temperature)
    temp_params = temp_def.init(temp_key)['params']
    temp = train_state.TrainState.create(
        apply_fn=temp_def.apply,
        params=temp_params,
        tx=optax.adam(learning_rate=config.temp_lr)
    )
    
    # Create the SAC state
    return SACState(
        actor=actor,
        critic=critic,
        target_critic_params=critic_params,  # Initial sync with critic
        temp=temp,
        rng=rng,
        step=0,
        config=config
    )

# --- SAC Agent Class ---

@struct.dataclass
class SACAgent(RLAgent):
    state: SACState

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        agent_config: Dict,
        **kwargs # Accept potential extra kwargs from base class, though SAC doesn't use them now
    ) -> "SACAgent":
        """Creates the SACAgent with its initial state."""
        # Use the existing factory function to create the underlying state
        
        agent_config["target_entropy"] = -actions.shape[-1] * agent_config["target_entropy_multiplier"]
        initial_state = create_sac_learner(
            seed=seed,
            observations=observations,
            actions=actions,
            config=SACConfig(**agent_config),
        )
        return cls(state=initial_state)

    def update(self, batch: Batch) -> Tuple["SACAgent", InfoDict]:
        """Performs a single SAC training step using the external JIT function."""
        new_state, info = _sac_update_step(self.state, batch)
        # Return a new agent instance wrapping the new state
        return SACAgent(state=new_state), info

    def sample(self, observation: jnp.ndarray) -> Tuple["SACAgent", jnp.ndarray]:
        """Samples an action stochastically using the external JIT function."""
        new_state, action = _sac_sample_step(self.state, observation)
        # Return a new agent instance wrapping the new state (with updated RNG)
        return SACAgent(state=new_state), action

    def sample_eval(self, observation: jnp.ndarray) -> Tuple["SACAgent", jnp.ndarray]:
        """Samples an action deterministically using the external JIT function."""
        action = _sac_sample_eval_step(self.state, observation)
        # Evaluation sampling doesn't change the agent's state (no RNG update)
        # Return self and the action
        return self, action



