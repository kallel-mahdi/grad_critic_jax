# td3_gc.py
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
from flax.training import train_state
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_map
import jax.flatten_util # Added for gradient flattening/unflattening

# Import networks
from networks import  DoubleCritic, DeterministicActor,MLP,default_init

# Import base TD3 components
from td3 import (
    TD3State,
    TD3Agent,
    TD3Config,
    target_update,
    update_critic as base_update_critic,
)

# Import common components from utils and base_agent
from utils import Batch, PRNGKey, Params, InfoDict
# Import composable actor updates
from actor_updates import update_td3gc_actor

# --- Config and State for TD3-GC ---

@struct.dataclass(kw_only=True)
class TD3ConfigGC(TD3Config):
    # Inherits non-default fields from TD3Config
    gamma_critic_lr: float 
    target_gamma_critic_update_period: int 
    share_critic_backbone: bool = False  # New option



@struct.dataclass
class TD3StateGC(TD3State):
    # Inherits all TD3State fields
    gamma_critic: train_state.TrainState  # Add gamma critic state
    target_gamma_critic_params: Params    # Add target gamma critic params
    config: TD3ConfigGC                   # Ensure config type hint is updated

# --- Gamma Critic Network (Copied from sac_gc.py) ---

class GammaCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray]
    num_params: int  # Output size for gamma parameters
    use_layer_norm: bool = False
    stop_backbone_grad: bool = False  # New flag

    def setup(self):
        # Main network excluding final layer
        self.feature_net = MLP(self.hidden_dims, activate_final=True, use_layer_norm=self.use_layer_norm)
        # Gamma parameter output head
        self.gamma_head = nn.Dense(self.num_params, kernel_init=default_init())

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        features = self.feature_net(inputs)
        
        # Stop gradient flow from features if sharing backbone
        if self.stop_backbone_grad:
            features = jax.lax.stop_gradient(features)
            
        # Output gamma parameters
        gamma_params = self.gamma_head(features)
        return gamma_params

class DoubleGammaCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    num_params: int = None  # Output size for gamma parameters, set during init
    use_layer_norm: bool = False
    stop_backbone_grad: bool = False  # New flag

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
            num_params=self.num_params,
            stop_backbone_grad=self.stop_backbone_grad  # Pass the flag
        )(states, actions)

        return gamma_params[0], gamma_params[1]  # Return the two sets of gamma parameters


# --- TD3-GC Gamma Critic Update ---

def update_gamma_critic(key_gamma: PRNGKey, state: TD3StateGC, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    """Updates the gamma critic network using per-sample TD3 actor gradients."""

    # 1. Define the TD3 actor loss function for a single observation, needed for per-sample gradients
    #    Note: We use the *current* critic params here, as per standard actor loss calculation.
    def single_actor_loss_fn(actor_params, critic_params, single_obs):
        # Ensure observation has batch dimension
        obs = jnp.expand_dims(single_obs, 0)
        # Get deterministic action from actor
        actions = state.actor.apply_fn({'params': actor_params}, obs)
        # Get Q-value from the first critic
        q1, _ = state.critic.apply_fn({'params': critic_params}, obs, actions)
        # TD3 actor loss: maximize Q-value
        loss = -q1.mean() # Mean over the single-element batch dim
        return loss

    # 2. Compute per-sample actor gradients w.r.t actor_params for *next* observations
    #    Vectorize the gradient calculation across the batch dimension of next_observations
    batch_grad_fn = jax.vmap(
        lambda obs: jax.grad(single_actor_loss_fn)(state.actor.params, state.critic.params, obs), # Grad w.r.t actor_params (arg 0)
        in_axes=0 # Vectorize over observations (first argument to lambda)
    )
    per_sample_grads = batch_grad_fn(batch.next_observations)

    # Flatten gradients to create concatenated representation (batch_size, num_actor_params)
    flat_grads, _ = jax.flatten_util.ravel_pytree(per_sample_grads)
    # Reshape required since ravel_pytree concatenates batch and param dims
    actor_param_count = state.gamma_critic.apply_fn.__self__.num_params # Get num_params from the module instance
    actor_next_grads = jnp.reshape(flat_grads, (batch.observations.shape[0], actor_param_count))


    # 3. Get next actions from the *target* actor (deterministic, no noise needed here)
    next_actions = state.actor.apply_fn({'params': state.target_actor_params}, batch.next_observations)
   

    # 4. Get gamma values for the next state-action pairs from the *target* gamma critic
    gamma1_next, gamma2_next = state.gamma_critic.apply_fn(
        {'params': state.target_gamma_critic_params},
        batch.next_observations,
        next_actions
    )
    gamma_next_target = (gamma1_next + gamma2_next) / 2.0 # Average the two target gammas

    # 5. Compute the Bellman target for gamma: discount * mask * (actor_grads + gamma_next_target)
    masks_expanded = jnp.expand_dims(batch.masks, axis=-1)  # Shape: (batch_size, 1)
    next_gamma_value = state.config.discount * masks_expanded * (actor_next_grads + gamma_next_target)
    next_gamma_value = jax.lax.stop_gradient(next_gamma_value) # Stop gradient flow

    # 6. Define loss function for the gamma critic update
    def gamma_loss_fn(gamma_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current gamma values for the batch observations and actions
        gamma1, gamma2 = state.gamma_critic.apply_fn(
            {'params': gamma_params},
            batch.observations,
            batch.actions
        )

        # Compute MSE loss against the Bellman target
        gamma_loss = ((gamma1 - next_gamma_value)**2 + (gamma2 - next_gamma_value)**2).mean()

        return gamma_loss, {
            'gamma_loss': gamma_loss,
            'gamma1_mean': gamma1.mean(),
            'gamma2_mean': gamma2.mean()
        }

    # 7. Compute gradients and update gamma critic
    grads, info = jax.grad(gamma_loss_fn, has_aux=True)(state.gamma_critic.params)
    new_gamma_critic = state.gamma_critic.apply_gradients(grads=grads)

    return new_gamma_critic, info


# --- TD3-GC Actor Update (with Gamma Correction) ---

def update_actor(state: TD3StateGC, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    """Updates the TD3 actor with gradient correction using composable system."""
    return update_td3gc_actor(state, batch)


# --- TD3-GC Update Step ---

@jax.jit
def _td3_gc_update_step(state: TD3StateGC, batch: Batch) -> Tuple[TD3StateGC, InfoDict]:
    """Single TD3-GC update step (external JIT)."""
    rng, key_critic, key_gamma = jax.random.split(state.rng, 3)

    # --- Critic Update ---
    # Use the base TD3 critic update function. It needs the target actor params.
    # Note: base_update_critic implicitly uses state.config
    new_critic, critic_info = base_update_critic(key_critic, state, batch)

    # Create a temporary state containing the updated critic for subsequent updates
    temp_state = state.replace(critic=new_critic)

    # --- Gamma Critic Update with Shared Backbone ---
    if state.config.share_critic_backbone:
        # Copy critic backbone to gamma critic before update
        updated_gamma_params = copy_critic_backbone_to_gamma_critic(
            new_critic.params, 
            state.gamma_critic.params
        )
        # Update gamma critic state with copied backbone
        temp_gamma_critic = state.gamma_critic.replace(params=updated_gamma_params)
        temp_state = temp_state.replace(gamma_critic=temp_gamma_critic)

    # --- Gamma Critic Update ---
    # Update gamma critic using the updated value critic state
    new_gamma_critic, gamma_info = update_gamma_critic(key_gamma, temp_state, batch)

    # Update the temporary state again
    temp_state = temp_state.replace(gamma_critic=new_gamma_critic)

    # --- Conditional Actor and Target Updates ---
    # Actor update, target actor update, and target critic update are delayed together
    # Target gamma critic update has its own frequency

    # Update Target Gamma Critic (conditional)
    def _update_target_gamma(state):
        # Use the 'temp_state' which has the new_gamma_critic params
        return target_update(state.gamma_critic.params, state.target_gamma_critic_params, state.config.tau)
    def _no_update_target_gamma(state):
        return state.target_gamma_critic_params

    new_target_gamma_params = jax.lax.cond(
        state.step % state.config.target_gamma_critic_update_period == 0,
        _update_target_gamma,
        _no_update_target_gamma,
        temp_state # Pass the state with the updated gamma critic
    )

    # Update Actor, Target Actor, Target Critic (conditional)
    def _update_actor_and_targets(state_for_actor_update):
        # Update actor using the corrected update function
        # state_for_actor_update already contains new_critic and new_gamma_critic
        new_actor, actor_info = update_actor(state_for_actor_update, batch)

        # Soft update target networks (actor and critic)
        new_target_critic_params = target_update(state_for_actor_update.critic.params, state_for_actor_update.target_critic_params, state_for_actor_update.config.tau)
        new_target_actor_params = target_update(new_actor.params, state_for_actor_update.target_actor_params, state_for_actor_update.config.tau)
        return new_actor, new_target_actor_params, new_target_critic_params, actor_info

    def _no_update_actor_and_targets(state_for_actor_update):
        # Keep actor and targets the same, return empty actor info
        actor_info = {'actor_loss': jnp.nan} # Or jnp.nan? Log appropriately outside.
        return state_for_actor_update.actor, state_for_actor_update.target_actor_params, state_for_actor_update.target_critic_params, actor_info

    # Use jax.lax.cond for conditional execution on device
    new_actor, new_target_actor_params, new_target_critic_params, actor_info = jax.lax.cond(
        state.step % state.config.policy_delay == 0,
        _update_actor_and_targets,
        _no_update_actor_and_targets,
        temp_state # Pass the temp_state containing new_critic and new_gamma_critic
    )

    # Create the final new state with all updated values
    new_state = state.replace(
        actor=new_actor,
        critic=new_critic,
        gamma_critic=new_gamma_critic,
        target_actor_params=new_target_actor_params,
        target_critic_params=new_target_critic_params,
        target_gamma_critic_params=new_target_gamma_params,
        rng=rng,
        step=state.step + 1
    )

    # Combine info dicts
    all_info = {**critic_info, **gamma_info, **actor_info}
    return new_state, all_info


# --- Factory Function for TD3-GC ---

def create_td3_gc_learner(
    seed: int,
    observations: jnp.ndarray, # Sample observation for initialization
    actions: jnp.ndarray,      # Sample action for initialization
    config: TD3ConfigGC,       # Use the GC config
) -> TD3StateGC:               # Return the GC state
    """Factory function to create TD3-GC state."""
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, gamma_critic_key = jax.random.split(rng, 4)

    action_dim = actions.shape[-1]

    # Initialize Actor network and state (same as TD3)
    actor_def = DeterministicActor(action_dim=action_dim, max_action=config.max_action,
                                   hidden_dims=config.hidden_dims)
    actor_params = actor_def.init(actor_key, observations)['params']
    actor = train_state.TrainState.create(
        apply_fn=actor_def.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=config.actor_lr)
    )

    # Initialize Critic network and state (same as TD3)
    critic_def = DoubleCritic(config.hidden_dims,use_layer_norm=config.use_layer_norm)
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = train_state.TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=config.critic_lr)
    )

    # Compute total number of parameters in actor network for gamma critic output
    actor_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_params))

    # Initialize GammaCritic network and state
    # Ensure activations are passed if needed, default is relu
    gamma_critic_def = DoubleGammaCritic(
        hidden_dims=config.hidden_dims, # Use same hidden dims as critic? Or specify separately? Using same for now.
        num_params=actor_param_count,
        use_layer_norm=config.use_layer_norm,
        stop_backbone_grad=config.share_critic_backbone  # Set flag
    )
    gamma_critic_params = gamma_critic_def.init(gamma_critic_key, observations, actions)['params']
    gamma_critic = train_state.TrainState.create(
        apply_fn=gamma_critic_def.apply,
        params=gamma_critic_params,
        tx=optax.adam(learning_rate=config.gamma_critic_lr) # Use specific LR
    )

    # Create the TD3-GC state
    return TD3StateGC(
        actor=actor,
        critic=critic,
        gamma_critic=gamma_critic,
        target_actor_params=actor_params, # Init targets same as online params
        target_critic_params=critic_params,
        target_gamma_critic_params=gamma_critic_params,
        rng=rng,
        step=0,
        config=config # Store the full GC config
    )


# --- TD3-GC Agent Class ---

@struct.dataclass
class TD3AgentGC(TD3Agent): # Inherit from base TD3Agent
    state: TD3StateGC         # Use the GC state type
    StateType: Any = TD3StateGC # Explicitly set state type if needed

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        agent_config: Dict,
        **kwargs # Accept potential extra kwargs
    ) -> "TD3AgentGC":
        """Creates the TD3AgentGC with its initial state."""
        # Create the specific TD3ConfigGC from the dictionary
        gc_config = TD3ConfigGC(**agent_config)

        # Use the GC learner factory function
        initial_state = create_td3_gc_learner(
            seed=seed,
            observations=observations,
            actions=actions,
            config=gc_config,
        )
        return cls(state=initial_state)

    def update(self, batch: Batch) -> Tuple["TD3AgentGC", InfoDict]:
        """Performs a single TD3-GC training step."""
        new_state, info = _td3_gc_update_step(self.state, batch)
        return TD3AgentGC(state=new_state), info

    # sample and sample_eval methods are inherited from TD3Agent and should work correctly
    # as they only depend on the actor state which has the same structure.

def copy_critic_backbone_to_gamma_critic(critic_params: Params, gamma_critic_params: Params) -> Params:
    """Copies critic backbone parameters to gamma critic backbone."""
    from flax import traverse_util
    
    # Flatten both parameter trees
    flat_critic = traverse_util.flatten_dict(critic_params, sep='/')
    flat_gamma = traverse_util.flatten_dict(gamma_critic_params, sep='/')
    
    # Copy feature_net parameters from critic to gamma critic
    # DoubleCritic structure: params -> [critic0, critic1] -> feature_net
    # DoubleGammaCritic structure: params -> [gamma0, gamma1] -> feature_net
    
    for key in flat_gamma:
        if 'feature_net' in key:
            # Direct copy from corresponding critic feature_net
            if key in flat_critic:
                flat_gamma[key] = flat_critic[key]  # Just copy, don't freeze
    
    # Unflatten (no need to freeze)
    return traverse_util.unflatten_dict(flat_gamma, sep='/')