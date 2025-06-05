from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
from flax.training import train_state
from flax import struct
import jax
import jax.numpy as jnp
import optax
import jax.flatten_util
from flax.core.frozen_dict import FrozenDict

# Import CrossQ components
from crossq import (
    CrossQState,
    CrossQAgent,
    CrossQConfig,
    BatchNormTrainState,
    StochasticActor,
    DoubleCritic,
    update_temperature,
)

# Import common utilities
from utils import Batch, BatchRenorm, default_init, PRNGKey, Params, InfoDict


@struct.dataclass
class CrossQConfigGC(CrossQConfig):
    pass  # No additional config needed since we removed gamma_critic_lr and target_gamma_critic_update_period


@struct.dataclass
class CrossQStateGC(CrossQState):
    gamma_critic: BatchNormTrainState  # Add gamma critic with BatchNorm support
    # Remove target_gamma_critic_params since CrossQ doesn't use target networks
    config: CrossQConfigGC


# --- Gamma Critic Network ---

class GammaCritic(nn.Module):
    hidden_dims: Sequence[int]
    num_params: int  # Output size for gamma parameters
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # Flatten observations if needed
        if len(observations.shape) > 2:
            observations = observations.reshape(observations.shape[0], -1)
        
        x = jnp.concatenate([observations, actions], -1)

        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        for n_units in self.hidden_dims:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
            
            if self.use_batch_norm:
                x = BatchRenorm(use_running_average=not train)(x)
            else:
                x_dummy = BatchRenorm(use_running_average=not train)(x)

        # Stop gradient flow from features
        #x = jax.lax.stop_gradient(x)
        
        # Gamma parameter output head
        gamma_params = nn.Dense(self.num_params)(x)
        return gamma_params

class DoubleGammaCritic(nn.Module):
    hidden_dims: Sequence[int]
    use_batch_norm: bool = True
    num_qs: int = 2
    num_params: int = None  # Output size for gamma parameters

    @nn.compact
    def __call__(self, states, actions, train: bool = False):
        # Creates multiple GammaCritic networks and runs them in parallel
        VmapGammaCritic = nn.vmap(
            GammaCritic,
            variable_axes={'params': 0, 'batch_stats': 0},  # Map over parameters and batch stats
            split_rngs={'params': True, 'batch_stats': True},  # Split RNGs for each critic
            in_axes=None,  # Inputs (states, actions) are shared
            out_axes=0,  # Stack outputs along the first axis
            axis_size=self.num_qs  # Number of critics to create
        )
        
        gamma_params = VmapGammaCritic(
            self.hidden_dims, 
            use_batch_norm=self.use_batch_norm,
            num_params=self.num_params
        )(states, actions, train)
        
        return gamma_params[0], gamma_params[1]  # Return the two sets of gamma parameters

# --- SAC Actor Update (with Gamma Correction) ---

def update_actor(key_actor: PRNGKey, state: CrossQStateGC, batch: Batch) -> Tuple[BatchNormTrainState, InfoDict]:
    # Get temperature value (scalar)
    temperature = state.temp.apply_fn({'params': state.temp.params})

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Apply actor model to get action distribution using actor's apply_fn
        variables = {'params': actor_params, 'batch_stats': state.actor.batch_stats}
        
        # Always use mutable=["batch_stats"] regardless of whether batch norm is used
        model_output, new_model_state = state.actor.apply_fn(
            variables, batch.observations, train=True, mutable=['batch_stats']
        )
            
        actions, log_probs = model_output.sample_and_log_prob(seed=key_actor)

        # Evaluate actions using the critic network's apply_fn
        critic_variables = {'params': state.critic.params, 'batch_stats': state.critic.batch_stats}
        
        # Always use mutable=False for critic during actor update (no training)
        q_values = state.critic.apply_fn(
            critic_variables, batch.observations, actions, train=False, mutable=False
        )
        
        # Take min across critics
        q = jnp.min(q_values, axis=0)

        # Actor loss: encourages actions with high Q-values and high entropy.
        actor_loss = (log_probs * temperature - q).mean()

        # Return loss, new batch_stats, and auxiliary info (entropy).
        return actor_loss, (new_model_state, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        })

    # Compute gradients and update actor model using TrainState's apply_gradients
    (actor_loss, (new_model_state, info)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)
    
    # Define function for gamma correction
    def _apply_gamma_correction(grads_and_state):
        grads, state = grads_and_state
        
        # Get actions from current policy
        variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
        dist = state.actor.apply_fn(variables, batch.observations, train=False, mutable=False)
        policy_actions, _ = dist.sample_and_log_prob(seed=key_actor)
        
        # Get gamma values for policy actions
        gamma_variables = {'params': state.gamma_critic.params, 'batch_stats': state.gamma_critic.batch_stats}
        gamma1_pi, gamma2_pi = state.gamma_critic.apply_fn(
            gamma_variables,
            batch.observations,
            policy_actions,
            train=False,
            mutable=False
        )

        # Get gamma values for batch actions
        gamma1_batch, gamma2_batch = state.gamma_critic.apply_fn(
            gamma_variables,
            batch.observations,
            batch.actions,
            train=False,
            mutable=False
        )
        
        # Compute average gamma values
        gamma_pi = (gamma1_pi + gamma2_pi) / 2.0
        gamma_batch = (gamma1_batch + gamma2_batch) / 2.0
        
        # Compute correction term and average over the batch dimension
        grad_correction = gamma_pi - gamma_batch
        mean_grad_correction = jnp.mean(grad_correction, axis=0) # Shape: (num_params,)
        
        # Flatten gradients and apply correction
        flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grads)
        

        corrected_flat_grads = flat_grads + mean_grad_correction
        
        # Unflatten corrected gradients back to the original pytree structure
        corrected_grads = unravel_fn(corrected_flat_grads)
        
        return corrected_grads

    corrected_grads = _apply_gamma_correction((grads, state))
    
    # # --- Compute Gradient Similarity Metrics ---
    # flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
    # flat_corrected_grads, _ = jax.flatten_util.ravel_pytree(corrected_grads)

    # norm_original = jnp.linalg.norm(flat_grads)
    # norm_corrected = jnp.linalg.norm(flat_corrected_grads)
    # dot_product = jnp.dot(flat_grads, flat_corrected_grads)

    # # Handle potential division by zero if norms are zero
    # cosine_similarity = jnp.where(
    #     (norm_original > 1e-8) & (norm_corrected > 1e-8), # Add epsilon for numerical stability
    #     dot_product / (norm_original * norm_corrected),
    #     0.0 # Define similarity as 0 if either gradient is zero
    # )
    # cosine_distance = 1.0 - cosine_similarity

    # info['original_grad_norm'] = norm_original
    # info['corrected_grad_norm'] = norm_corrected
    # info['grad_cosine_similarity'] = cosine_similarity
    # info['grad_cosine_distance'] = cosine_distance
    # --- End Gradient Similarity Metrics ---

    # Update actor
    new_actor = state.actor.apply_gradients(grads=corrected_grads)
    # Update batch_stats with new values - converting to FrozenDict
    new_actor = new_actor.replace(batch_stats=new_model_state['batch_stats'])

    return new_actor, info


# Add Gamma Critic Update
def update_gamma_critic(key_gamma: PRNGKey, state: CrossQStateGC, batch: Batch) -> Tuple[BatchNormTrainState, InfoDict]:
    """Updates the gamma critic network using per-sample actor gradients with joint forward pass."""
    
    # 1. Get next state actions from the policy
    variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
    dist = state.actor.apply_fn(variables, batch.next_observations, train=False, mutable=False)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key_gamma)
    
    # 2. Compute per-sample actor gradients for next observations
    temperature = state.temp.apply_fn({'params': state.temp.params})
    
    # Function to compute actor loss gradient for a single observation
    def actor_loss_fn(params, single_obs):
        # Ensure observation has batch dimension
        obs = jnp.expand_dims(single_obs, 0)
        
        # Get action distribution
        variables = {'params': params, 'batch_stats': state.actor.batch_stats}
        dist = state.actor.apply_fn(variables, obs, train=False, mutable=False)
        actions, log_probs = dist.sample_and_log_prob(seed=key_gamma)
        
        # Get Q-values
        critic_variables = {'params': state.critic.params, 'batch_stats': state.critic.batch_stats}
        q_values = state.critic.apply_fn(critic_variables, obs, actions, train=False, mutable=False)
        min_q = jnp.min(q_values, axis=0)
        
        # Actor loss
        return (temperature * log_probs - min_q).mean()
    
    # Vectorize gradient computation across batch
    batch_grad_fn = jax.vmap(
        lambda obs: jax.grad(actor_loss_fn)(state.actor.params, obs),
        in_axes=0
    )
    
    # Compute gradients for each observation
    per_sample_grads = batch_grad_fn(batch.next_observations)
    
    # Flatten gradients to create concatenated representation
    flat_grads = []
    for leaf in jax.tree_util.tree_leaves(per_sample_grads):
        flat_grads.append(jnp.reshape(leaf, (leaf.shape[0], -1)))
    
    actor_next_grads = jnp.concatenate(flat_grads, axis=1)
    
    # 3. Get gamma values for next state (no target network in CrossQ)
    gamma_variables = {'params': state.gamma_critic.params, 'batch_stats': state.gamma_critic.batch_stats}
    gamma1_next, gamma2_next = state.gamma_critic.apply_fn(
        gamma_variables,
        batch.next_observations,
        next_actions,
        train=False,
        mutable=False
    )
    
    # 4. Compute target: (1-done) * discount * (actor_grads + next_gamma)
    gamma_next_target = (gamma1_next + gamma2_next) / 2.0
    
    masks_expanded = jnp.expand_dims(batch.masks, axis=-1)  # Shape: (256, 1)
    next_gamma_value = state.config.discount * masks_expanded * (actor_next_grads + gamma_next_target)
    
    # Stop gradient through target
    next_gamma_value = jax.lax.stop_gradient(next_gamma_value)
    
    # The key innovation: process both current and next observation-action pairs through the gamma critic
    # This ensures BatchRenorm statistics are calculated on the joint distribution
    joint_observations = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
    joint_actions = jnp.concatenate([batch.actions, next_actions], axis=0)
    
    def gamma_loss_fn(gamma_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Joint forward pass through gamma critic with both current and next pairs
        variables = {'params': gamma_params, 'batch_stats': state.gamma_critic.batch_stats}
        
        # Always use mutable=["batch_stats"] regardless of whether batch norm is used
        joint_gamma_values, new_model_state = state.gamma_critic.apply_fn(
            variables, joint_observations, joint_actions, train=True, mutable=['batch_stats']
        )
        
        # Split back into current and next gamma values
        batch_size = batch.observations.shape[0]
        current_gamma_values, _ = joint_gamma_values[0][:batch_size], joint_gamma_values[0][batch_size:]
        current_gamma2_values, _ = joint_gamma_values[1][:batch_size], joint_gamma_values[1][batch_size:]
        
        # 6. Compute MSE loss
        gamma_loss = ((current_gamma_values - next_gamma_value)**2).mean() + ((current_gamma2_values - next_gamma_value)**2).mean()
        
        return gamma_loss, (new_model_state, {
            'gamma_loss': gamma_loss,
            'gamma1_mean': current_gamma_values.mean(),
            'gamma2_mean': current_gamma2_values.mean()
        })
    
    # 7. Compute gradients and update gamma critic
    (gamma_loss, (new_model_state, info)), grads = jax.value_and_grad(gamma_loss_fn, has_aux=True)(state.gamma_critic.params)
    
    # Update gamma critic
    new_gamma_critic = state.gamma_critic.apply_gradients(grads=grads)
    # Update batch_stats with new values - converting to FrozenDict
    new_gamma_critic = new_gamma_critic.replace(batch_stats=new_model_state['batch_stats'])
    
    return new_gamma_critic, info


# --- SAC Update Steps (External JIT functions) ---

@jax.jit
def _crossq_gc_update_step(state: CrossQStateGC, batch: Batch) -> Tuple[CrossQStateGC, InfoDict]:
    """Single update step for CrossQ-GC (external JIT)."""
    rng, key_critic, key_actor, key_gamma = jax.random.split(state.rng, 4)

    # Update critic (reuse from CrossQ)
    from crossq import update_critic
    new_critic, critic_info = update_critic(key_critic, state, batch)

    # Update gamma critic (always update, no delay)
    # Create a temporary state with updated critic for gamma update
    temp_state_for_gamma = state.replace(critic=new_critic)
    new_gamma_critic, gamma_info = update_gamma_critic(key_gamma, temp_state_for_gamma, batch)

    # Update actor with policy delay
    should_update_actor = (state.step % state.config.policy_delay == 0)
    
    def _update_actor():
        # Update actor using updated critic and gamma critic
        temp_state = state.replace(critic=new_critic, gamma_critic=new_gamma_critic)
        new_actor, actor_info = update_actor(key_actor, temp_state, batch)
        
        # Update temperature using updated actor entropy
        temp_state_for_temp = temp_state.replace(actor=new_actor)
        new_temp, alpha_info = update_temperature(temp_state_for_temp, actor_info['entropy'])
        
        return new_actor, new_temp, {**actor_info, **alpha_info}
    
    def _no_update_actor():
        return state.actor, state.temp, {'actor_loss': jnp.nan, 'entropy': jnp.nan, 'temperature': jnp.nan, 'temp_loss': jnp.nan}
    
    new_actor, new_temp, actor_temp_info = jax.lax.cond(
        should_update_actor,
        lambda: _update_actor(),
        lambda: _no_update_actor()
    )

    # Create final new state with updated values (no target networks)
    new_state = state.replace(
        actor=new_actor,
        critic=new_critic,
        gamma_critic=new_gamma_critic,
        temp=new_temp,
        rng=rng,
        step=state.step + 1
    )

    # Combine all info dicts
    all_info = {**critic_info, **gamma_info, **actor_temp_info}
    return new_state, all_info




# --- SAC Learner (with Gamma Critic) ---

def create_crossq_gc_learner(
    seed: int,
    observations: jnp.ndarray,  # Sample observation for initialization
    actions: jnp.ndarray,       # Sample action for initialization
    config: CrossQConfigGC,
) -> CrossQStateGC:
    """Factory function to create CrossQ-GC state."""
    
    # Initialize PRNG keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, gamma_critic_key, temp_key = jax.random.split(rng, 5)
    
    # Initialize Actor network and state
    actor_def = StochasticActor(
        config.hidden_dims,
        actions.shape[-1],
        max_action=config.max_action,
        log_std_min=config.policy_log_std_min,
        log_std_max=config.policy_log_std_max,
        use_batch_norm=config.use_batch_norm,
    )
    
    # Initialize actor
    actor_variables = actor_def.init(actor_key, observations, train=False)
    # Create BatchNormTrainState for actor
    actor = BatchNormTrainState.create(
        apply_fn=actor_def.apply,
        params=actor_variables['params'],
        batch_stats=actor_variables.get('batch_stats', FrozenDict({})),
        tx=optax.adam(learning_rate=config.actor_lr)
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
        tx=optax.adam(learning_rate=config.critic_lr)
    )
    
    # Compute total number of parameters in actor network
    actor_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_variables['params']))
    
    # Initialize GammaCritic network and state
    gamma_critic_def = DoubleGammaCritic(
        config.hidden_dims,
        use_batch_norm=config.use_batch_norm,
        num_params=actor_param_count
    )
    
    # Initialize gamma critic
    gamma_critic_variables = gamma_critic_def.init(gamma_critic_key, observations, actions, train=False)
    
    # Create BatchNormTrainState for gamma critic
    gamma_critic = BatchNormTrainState.create(
        apply_fn=gamma_critic_def.apply,
        params=gamma_critic_variables['params'],
        batch_stats=gamma_critic_variables.get('batch_stats', FrozenDict({})),
        tx=optax.adam(learning_rate=config.critic_lr)  # Use same lr as critic
    )
    
    # Initialize Temperature parameter and state
    from networks import Temperature
    temp_def = Temperature(config.init_temperature)
    temp_params = temp_def.init(temp_key)['params']
    temp = train_state.TrainState.create(
        apply_fn=temp_def.apply,
        params=temp_params,
        tx=optax.adam(learning_rate=config.temp_lr)
    )
    
    # Create the CrossQ state with gamma critic (no target networks)
    return CrossQStateGC(
        actor=actor,
        critic=critic,
        gamma_critic=gamma_critic,
        temp=temp,
        rng=rng,
        step=0,
        config=config
    )


# --- SAC Agent Class (with Gamma Critic) ---

@struct.dataclass
class CrossQAgentGC(CrossQAgent):
    state: CrossQStateGC
    StateType: Any = CrossQStateGC # Explicitly set state type

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        agent_config: Dict,
        **kwargs # Accept potential extra kwargs from base class, though SAC doesn't use them now
    ) -> "CrossQAgent":
        """Creates the SACAgent with its initial state."""
        # Use the existing factory function to create the underlying state
        
        agent_config["target_entropy"] = -actions.shape[-1] * agent_config["target_entropy_multiplier"]
        initial_state = create_crossq_gc_learner(
            seed=seed,
            observations=observations,
            actions=actions,
            config=CrossQConfigGC(**agent_config),
        )
        return cls(state=initial_state)
    
    def update(self, batch: Batch) -> Tuple["CrossQAgentGC", InfoDict]:
        """Performs a single CrossQ-GC training step."""
        new_state, info = _crossq_gc_update_step(self.state, batch)
        return CrossQAgentGC(state=new_state), info

    def sample(self, observation: jnp.ndarray) -> Tuple["CrossQAgentGC", jnp.ndarray]:
        """Samples an action stochastically using the external JIT function."""
        from crossq import _crossq_sample_step
        new_state, action = _crossq_sample_step(self.state, observation)
        # Return a new agent instance wrapping the new state (with updated RNG)
        return CrossQAgentGC(state=new_state), action

    def sample_eval(self, observation: jnp.ndarray) -> Tuple["CrossQAgentGC", jnp.ndarray]:
        """Samples an action deterministically using the external JIT function."""
        from crossq import _crossq_sample_eval_step
        action = _crossq_sample_eval_step(self.state, observation)
        # Evaluation sampling doesn't change the agent's state (no RNG update)
        return self, action

    