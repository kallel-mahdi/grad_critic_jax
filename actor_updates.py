# actor_updates.py - Composable actor update components
from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.flatten_util
from flax.training import train_state
from utils import Batch, PRNGKey, Params, InfoDict


# --- Composable Loss Functions ---

def td3_loss_fn(state, batch: Batch) -> Callable[[Params], Tuple[jnp.ndarray, InfoDict]]:
    """Standard TD3 actor loss function."""
    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = state.actor.apply_fn({'params': actor_params}, batch.observations)
        q1, _ = state.critic.apply_fn({'params': state.critic.params}, batch.observations, actions)
        actor_loss = -(batch.discounts * q1).mean()
        #actor_loss = -(batch.discounts * q1).sum()/ (batch.discounts.sum())
        return actor_loss, {'actor_loss': actor_loss}
    return loss_fn


def td3bc_loss_fn(state, batch: Batch) -> Callable[[Params], Tuple[jnp.ndarray, InfoDict]]:
    """TD3-BC actor loss function (TD3 + behavior cloning)."""
    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = state.actor.apply_fn({'params': actor_params}, batch.observations)
        q1, _ = state.critic.apply_fn({'params': state.critic.params}, batch.observations, actions)
        
        # TD3 loss
        td3_loss = -q1.mean()
        
        # BC loss
        bc_loss = jnp.square(actions - batch.actions).mean()
        
        # Adaptive weighting
        mean_abs_q = jax.lax.stop_gradient(jnp.abs(q1).mean())
        loss_lambda = state.config.alpha / mean_abs_q
        
        # Combined loss
        actor_loss = td3_loss * loss_lambda + bc_loss
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'td3_loss': td3_loss, 
            'bc_loss': bc_loss,
            'loss_lambda': loss_lambda
        }
    return loss_fn


def sac_loss_fn(state, batch: Batch, key: PRNGKey) -> Callable[[Params], Tuple[jnp.ndarray, InfoDict]]:
    """SAC actor loss function."""
    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = state.actor.apply_fn({'params': actor_params}, batch.observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        
        q1, q2 = state.critic.apply_fn({'params': state.critic.params}, batch.observations, actions)
        q = jnp.minimum(q1, q2)
        
        temperature = state.temp.apply_fn({'params': state.temp.params})
        actor_loss = (log_probs * temperature - q).mean()
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }
    return loss_fn


# --- Gradient Correction Functions ---

def apply_gamma_correction(
    grads: Params, 
    state, 
    batch: Batch
) -> Tuple[Params, InfoDict]:
    """Apply TD3-GC style gradient correction."""
    # Get policy and batch actions
    policy_actions = state.actor.apply_fn({'params': state.actor.params}, batch.observations)
    
    # Get gamma values
    gamma1_pi, gamma2_pi = state.gamma_critic.apply_fn(
        {'params': state.gamma_critic.params}, batch.observations, policy_actions
    )
    gamma1_batch, gamma2_batch = state.gamma_critic.apply_fn(
        {'params': state.gamma_critic.params}, batch.observations, batch.actions
    )
    
    # Compute correction
    gamma_pi = (gamma1_pi + gamma2_pi) / 2.0
    gamma_batch = (gamma1_batch + gamma2_batch) / 2.0
    grad_correction = gamma_pi - gamma_batch
    mean_grad_correction = jnp.mean(grad_correction, axis=0)
    
    # Apply correction
    flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grads)
    corrected_flat_grads = flat_grads + mean_grad_correction
    corrected_grads = unravel_fn(corrected_flat_grads)
    
    # Compute metrics
    norm_original = jnp.linalg.norm(flat_grads)
    norm_corrected = jnp.linalg.norm(corrected_flat_grads)
    dot_product = jnp.dot(flat_grads, corrected_flat_grads)
    
    cosine_similarity = jnp.where(
        (norm_original > 1e-8) & (norm_corrected > 1e-8),
        dot_product / (norm_original * norm_corrected),
        0.0
    )
    
    correction_info = {
        'original_grad_norm': norm_original,
        'corrected_grad_norm': norm_corrected,
        'grad_cosine_similarity': cosine_similarity,
        'grad_cosine_distance': 1.0 - cosine_similarity
    }
    
    return corrected_grads, correction_info


def no_correction(grads: Params, state, batch: Batch) -> Tuple[Params, InfoDict]:
    """Identity function - no gradient correction."""
    return grads, {}


# --- Generic Actor Update Function ---

def update_actor_generic(
    state,
    batch: Batch,
    loss_fn: Callable[[Params], Tuple[jnp.ndarray, InfoDict]],
    correction_fn: Callable[[Params, Any, Batch], Tuple[Params, InfoDict]] = no_correction,
) -> Tuple[train_state.TrainState, InfoDict]:
    """Generic actor update with configurable loss and correction functions."""
    
    # Compute gradients using the provided loss function
    grads, loss_info = jax.grad(loss_fn, has_aux=True)(state.actor.params)
    
    # Apply gradient correction if provided
    corrected_grads, correction_info = correction_fn(grads, state, batch)
    
    # Update actor
    new_actor = state.actor.apply_gradients(grads=corrected_grads)
    
    # Combine info
    all_info = {**loss_info, **correction_info}
    
    return new_actor, all_info


# --- Convenience Functions for Each Algorithm ---

def update_td3_actor(state, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    """TD3 actor update."""
    return update_actor_generic(state, batch, td3_loss_fn(state, batch))


def update_td3bc_actor(state, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    """TD3-BC actor update."""
    return update_actor_generic(state, batch, td3bc_loss_fn(state, batch))


def update_td3gc_actor(state, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    """TD3-GC actor update.""" 
    return update_actor_generic(
        state, batch, td3_loss_fn(state, batch), apply_gamma_correction
    )


def update_td3gcbc_actor(state, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:
    """TD3-GCBC actor update (BC loss + gamma correction)."""
    return update_actor_generic(
        state, batch, td3bc_loss_fn(state, batch), apply_gamma_correction
    )


def update_sac_actor(state, batch: Batch, key: PRNGKey) -> Tuple[train_state.TrainState, InfoDict]:
    """SAC actor update."""
    return update_actor_generic(state, batch, sac_loss_fn(state, batch, key))


def update_sacgc_actor(state, batch: Batch, key: PRNGKey) -> Tuple[train_state.TrainState, InfoDict]:
    """SAC-GC actor update."""
    return update_actor_generic(
        state, batch, sac_loss_fn(state, batch, key), apply_gamma_correction
    ) 