from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
from flax.training import train_state
from flax import struct
import jax
import jax.numpy as jnp
import optax
import jax.flatten_util


# Import base SAC components
from sac import (
    SACState,
    SACAgent,
    SACConfig ,
    NormalTanhPolicy,
    DoubleCritic,
    Temperature,
    update_temperature,
    target_update,
    update_critic as base_update_critic,
)

# Import common utilities
from utils import Batch, MLP, default_init, PRNGKey, Params, InfoDict


@struct.dataclass
class SACConfigGC(SACConfig):
    gamma_critic_lr: float
    target_gamma_critic_update_period: int


@struct.dataclass
class SACStateGC(SACState): # Rename to avoid clash, inherit from BaseSACState
    gamma_critic: train_state.TrainState  # Add gamma critic
    target_gamma_critic_params: Params  # Add target gamma critic params
    # Ensure config type hint is updated
    config: SACConfigGC # Default to None, will be set in factory


# --- Gamma Critic Network ---

class GammaCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray]
    num_params: int  # Output size for gamma parameters

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

# --- SAC Actor Update (with Gamma Correction) ---

def update_actor(key_actor: PRNGKey, state: SACStateGC, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
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
        mean_grad_correction = jnp.mean(grad_correction, axis=0) # Shape: (num_params,)
        
        # Flatten gradients and apply correction
        flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grads)
        
        # Ensure correction has the same total size as flattened gradients
        if flat_grads.size != mean_grad_correction.size:
             raise ValueError(f"Flattened gradient size ({flat_grads.size}) != "
                              f"Correction size ({mean_grad_correction.size})")

        corrected_flat_grads = flat_grads + mean_grad_correction
        
        # Unflatten corrected gradients back to the original pytree structure
        corrected_grads = unravel_fn(corrected_flat_grads)
        
        return corrected_grads

   

    corrected_grads = _apply_gamma_correction((grads, state))
    
    # --- Compute Gradient Similarity Metrics ---
    flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
    flat_corrected_grads, _ = jax.flatten_util.ravel_pytree(corrected_grads)

    norm_original = jnp.linalg.norm(flat_grads)
    norm_corrected = jnp.linalg.norm(flat_corrected_grads)
    dot_product = jnp.dot(flat_grads, flat_corrected_grads)

    # Handle potential division by zero if norms are zero
    cosine_similarity = jnp.where(
        (norm_original > 1e-8) & (norm_corrected > 1e-8), # Add epsilon for numerical stability
        dot_product / (norm_original * norm_corrected),
        0.0 # Define similarity as 0 if either gradient is zero
    )
    cosine_distance = 1.0 - cosine_similarity

    info['original_grad_norm'] = norm_original
    info['corrected_grad_norm'] = norm_corrected
    info['grad_cosine_similarity'] = cosine_similarity
    info['grad_cosine_distance'] = cosine_distance
    # --- End Gradient Similarity Metrics ---

    new_actor = state.actor.apply_gradients(grads=corrected_grads)

    return new_actor, info


# Add Gamma Critic Update
def update_gamma_critic(key_gamma: PRNGKey, state: SACStateGC, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
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


# --- SAC Update Steps (External JIT functions) ---

@jax.jit
def _sac_gc_update_step(state: SACStateGC, batch: Batch) -> Tuple[SACStateGC, InfoDict]:
    """Single update step for GC-SAC (external JIT)."""
    rng, key_critic, key_actor, key_gamma = jax.random.split(state.rng, 4)

    # Update critic (using base update function)
    # Note: base_update_critic expects the base SACState structure within the GC state
    new_critic, critic_info = base_update_critic(key_critic, state, batch)

    # Conditionally update target critic parameters
    def _update_target_critic(state_and_new_critic):
        state, new_critic = state_and_new_critic
        return target_update(new_critic.params, state.target_critic_params, state.config.tau)

    def _no_update_target_critic(state_and_new_critic):
        state, _ = state_and_new_critic
        return state.target_critic_params

    new_target_critic_params = jax.lax.cond(
        state.step % state.config.target_update_period == 0,
        _update_target_critic,
        _no_update_target_critic,
        (state, new_critic)
    )

    # Update gamma critic
    # Create a temporary state with updated critic for gamma update
    temp_state_for_gamma = state.replace(critic=new_critic)
    new_gamma_critic, gamma_info = update_gamma_critic(key_gamma, temp_state_for_gamma, batch)

    # Conditionally update target gamma critic parameters
    def _update_target_gamma(state_and_new_gamma):
        state, new_gamma = state_and_new_gamma
        return target_update(new_gamma.params, state.target_gamma_critic_params, state.config.tau)

    def _no_update_target_gamma(state_and_new_gamma):
        state, _ = state_and_new_gamma
        return state.target_gamma_critic_params

    # Target update happens based on the *new* gamma critic state but *old* target params
    new_target_gamma_params = jax.lax.cond(
        state.step % state.config.target_update_period == 0,
        _update_target_gamma,
        _no_update_target_gamma,
        (state, new_gamma_critic) # Pass state for config.tau and old target, new_gamma for new params
    )

    # Update actor (with gamma correction)
    # Create a temporary state with updated critic and gamma critic for actor update
    temp_state_for_actor = state.replace(
        critic=new_critic,
        gamma_critic=new_gamma_critic
    )
    new_actor, actor_info = update_actor(key_actor, temp_state_for_actor, batch)

    # Update temperature
    # Temperature update depends on actor's entropy, use state containing new actor
    temp_state_for_temp = temp_state_for_actor.replace(actor=new_actor)
    new_temp, alpha_info = update_temperature(temp_state_for_temp, actor_info['entropy'])

    # Create final new state with updated values
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

    # Combine all info dicts, handling potential key collisions if necessary
    # (Assuming no collisions for now)
    all_info = {**critic_info, **gamma_info, **actor_info, **alpha_info}
    return new_state, all_info




# --- SAC Learner (with Gamma Critic) ---

def create_sac_gc_learner(
    seed: int,
    observations: jnp.ndarray,  # Sample observation for initialization
    actions: jnp.ndarray,       # Sample action for initialization
    config: SACConfig, # Keep using SACConfig
    policy_final_fc_init_scale: float = 1.0
) -> SACStateGC:
    """Factory function to create SAC state with gamma critic."""
    
    action_dim = actions.shape[-1]
   
    # Initialize PRNG keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, gamma_critic_key, temp_key = jax.random.split(rng, 5)
    
    # Initialize Actor network and state
    actor_def = NormalTanhPolicy(
        config.hidden_dims,
        action_dim,
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
    
    
    # Compute total number of parameters in actor network
    actor_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_params))
    
    # Initialize GammaCritic network and state
    gamma_critic_def = DoubleGammaCritic(config.hidden_dims,num_params=actor_param_count)
    gamma_critic_params = gamma_critic_def.init(gamma_critic_key, observations, actions)['params']
    gamma_critic = train_state.TrainState.create(
        apply_fn=gamma_critic_def.apply,
        params=gamma_critic_params,
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
    
    # Create the SAC state with gamma critic
    return SACStateGC(
        actor=actor,
        critic=critic,
        target_critic_params=critic_params,
        gamma_critic=gamma_critic,
        target_gamma_critic_params=gamma_critic_params,
        temp=temp,
        rng=rng,
        step=0,
        config=config # Pass the full config object
    )


# --- SAC Agent Class (with Gamma Critic) ---

@struct.dataclass
class SACAgentGC(SACAgent):
    state: SACStateGC
    StateType: Any = SACStateGC # Explicitly set state type

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
        initial_state = create_sac_gc_learner(
            seed=seed,
            observations=observations,
            actions=actions,
            config=SACConfigGC(**agent_config),
        )
        return cls(state=initial_state)
    
    def update(self, batch: Batch) -> Tuple["SACAgentGC", InfoDict]:
        """Performs a single GC-SAC training step."""
        new_state, info = _sac_gc_update_step(self.state, batch)
        return SACAgentGC(state=new_state), info

    