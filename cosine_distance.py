# cosine_distance.py
from typing import Tuple, Dict, Any
import copy
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import wandb
from tqdm import tqdm
from functools import partial
from td3_gc import TD3AgentGC, TD3StateGC, create_td3_gc_learner, TD3ConfigGC
from td3 import update_critic as base_update_critic
from networks import DoubleCritic
from utils import Batch, ReplayBuffer, PRNGKey, Params, InfoDict
from actor_updates import td3_loss_fn, apply_gamma_correction
import flax.linen as nn
from flax.training import train_state
import optax
import jax.flatten_util
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

jit_update_critic = jax.jit(base_update_critic)

#TODO: Add discount

def collect_rollouts(
    agent: TD3AgentGC,
    env: gym.Env,
    num_steps: int = 100_000,
    deterministic: bool = True,
    num_parallel_envs: int = 8,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect rollouts using the current policy for training true gradient critic.
    Uses vectorized environments for much faster data collection.
    
    Args:
        agent: Current TD3-GC agent
        env: Single environment (used to get env_id for creating vectorized envs)
        num_steps: Total number of steps to collect
        deterministic: Whether to use deterministic policy (True for true gradient)
        num_parallel_envs: Number of parallel environments to use
        
    Returns:
        Tuple of (observations, actions, rewards, next_observations, masks)
    """
    import gymnasium as gym
    
    # Create vectorized environment
    # Extract environment ID from the single environment
    
    env_id = env.spec.id
    
    vec_env = gym.make_vec(
    env_id,
    num_envs=num_parallel_envs,
    # vectorization_mode="async",
    # vector_kwargs={"context": "spawn"} 
    vectorization_mode="sync"
    )
    print(f"Created vectorized environment with {num_parallel_envs} parallel environments")

    

    observations = []
    actions = []
    rewards = []
    next_observations = []
    masks = []
    discounts = []
    # Reset all environments
    obs, _ = vec_env.reset()
    steps_collected = 0
    
    # Calculate steps per environment
    steps_per_env = num_steps // num_parallel_envs
    total_steps_needed = steps_per_env * num_parallel_envs
    
    progress_bar = tqdm(total=total_steps_needed, desc="Collecting parallel rollouts")
    
    discount = jnp.ones(num_parallel_envs)
    
    while steps_collected < total_steps_needed:
        # obs is shape (num_envs, obs_dim)
        if deterministic:
            # Use evaluation sampling (deterministic)
            _, action = agent.sample_eval(obs)
        else:
            # Use stochastic sampling
            agent, action = agent.sample(obs)
        
        # action is shape (num_envs, action_dim)
        # Execute actions in all environments
        next_obs, reward, terminated, truncated, _ = vec_env.step(action)
        done = terminated | truncated  # Element-wise OR for boolean arrays
        mask = 1.0 - terminated.astype(np.float32)  # Only terminated, not truncated
        
        
        
        
        # Store transitions for all environments
        observations.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward.copy())
        next_observations.append(next_obs.copy())
        masks.append(mask.copy())
        discounts.append(discount.copy())
        
        discount = discount * gamma
        
        if any(done): discount.at[done==True].set(1.)
        
        obs = next_obs
        steps_collected += num_parallel_envs
        progress_bar.update(num_parallel_envs)

    progress_bar.close()
    vec_env.close()
        
    # Flatten the collected data
    # Each list element has shape (num_envs, ...), so we need to reshape
    observations = np.concatenate(observations, axis=0)  # (total_steps, obs_dim)
    actions = np.concatenate(actions, axis=0)  # (total_steps, action_dim)
    rewards = np.concatenate(rewards, axis=0)  # (total_steps,)
    next_observations = np.concatenate(next_observations, axis=0)  # (total_steps, obs_dim)
    masks = np.concatenate(masks, axis=0)  # (total_steps,)
    discounts = np.concatenate(discounts, axis=0)  # (total_steps,)
    print(f"Collected {observations.shape[0]} transitions using {num_parallel_envs} parallel environments")
    
    return (
        observations.astype(np.float32),
        actions.astype(np.float32),
        rewards.astype(np.float32),
        next_observations.astype(np.float32),
        masks.astype(np.float32),
        discounts.astype(np.float32)
    )
    

#@jax.jit
def _fresh_critic_update_step(
    temp_state: TD3StateGC,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    masks: jnp.ndarray,
    rng: PRNGKey,
    batch_size: int,
    dataset_size: int
) -> Tuple[TD3StateGC, PRNGKey]:
    """
    JIT-compiled single training step for fresh critic using existing td3.py logic.
    """
    rng, batch_rng, noise_rng = jax.random.split(rng, 3)
    
    # Sample batch
    batch_idx = jax.random.randint(batch_rng, (batch_size,), 0, dataset_size)
    batch_obs = observations[batch_idx]
    batch_act = actions[batch_idx]
    batch_rew = rewards[batch_idx]
    batch_next_obs = next_observations[batch_idx]
    batch_masks = masks[batch_idx]
    
    # Create batch object
    batch = Batch(
        observations=batch_obs,
        actions=batch_act,
        rewards=batch_rew,
        next_observations=batch_next_obs,
        masks=batch_masks,
        discounts=jnp.ones_like(batch_rew)  # Not used in critic update
    )
    
    # Update critic using existing td3.py logic
    new_critic, _ = jit_update_critic(noise_rng, temp_state, batch)
    new_temp_state = temp_state.replace(critic=new_critic, rng=rng)
    
    return new_temp_state, rng


def train_fresh_critic(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    masks: jnp.ndarray,
    agent_state: TD3StateGC,
    seed: int = 42,
    num_training_steps: int = 50_000
) -> train_state.TrainState:
    """
    Train a fresh critic from scratch using existing td3.py infrastructure.
    """
    rng = jax.random.PRNGKey(seed)
    rng, critic_key = jax.random.split(rng)
    
    # Create a deep copy of the agent state to avoid modifying the original
    temp_state = copy.deepcopy(agent_state)
    
    # # Initialize fresh critic with same architecture as original
    # critic_def = DoubleCritic(temp_state.config.hidden_dims, use_layer_norm=temp_state.config.use_layer_norm)
    # sample_obs = jnp.expand_dims(observations[0], 0)
    # sample_act = jnp.expand_dims(actions[0], 0)
    # critic_params = critic_def.init(critic_key, sample_obs, sample_act)['params']
    
    # fresh_critic = train_state.TrainState.create(
    #     apply_fn=critic_def.apply,
    #     params=critic_params,
    #     tx=optax.adam(learning_rate=temp_state.config.critic_lr)
    # )
    
    # # Replace the critic in temp_state with fresh critic
    # temp_state = temp_state.replace(
    #     critic=fresh_critic,
    #     target_critic_params=critic_params,  # Fresh target params too
    #     rng=rng
    # )
    
    # Convert data to JAX arrays
    observations = jax.device_put(observations)
    actions = jax.device_put(actions)
    rewards = jax.device_put(rewards)
    next_observations = jax.device_put(next_observations)
    masks = jax.device_put(masks)
    
    # Training parameters
    dataset_size = observations.shape[0]
    batch_size = min(256, dataset_size // 10)
    
    print(f"Training fresh critic with JIT compilation for {num_training_steps} steps...")
    
    # Training loop - JIT compiled step function
    for step in tqdm(range(num_training_steps), desc="Training fresh critic (JIT)"):
        temp_state, rng = _fresh_critic_update_step(
            temp_state,
            observations,
            actions,
            rewards,
            next_observations,
            masks,
            rng,
            batch_size,
            dataset_size
        )
    
    print("Fresh critic training completed!")
    return temp_state.critic


def compute_actor_gradient(
    state_or_critic: Any,
    batch: Batch,
    is_state: bool = True
) -> jnp.ndarray:
    """
    Compute actor gradient given critic and batch.
    
    Args:
        state_or_critic: Either TD3StateGC or just a critic TrainState
        batch: Batch of data
        is_state: Whether first arg is full state (True) or just critic (False)
        
    Returns:
        Flattened actor gradient
    """
    if is_state:
        # Using full TD3StateGC
        state = state_or_critic
        actor_params = state.actor.params
        actor_apply_fn = state.actor.apply_fn
        critic_params = state.critic.params
        critic_apply_fn = state.critic.apply_fn
    else:
        # Using just critic, need to get actor from somewhere else
        raise ValueError("Must provide full state for actor gradient computation")
    
    # Define actor loss function (same as TD3)
    def actor_loss_fn(actor_params: Params) -> jnp.ndarray:
        actions = actor_apply_fn({'params': actor_params}, batch.observations)
        q1, _ = critic_apply_fn({'params': critic_params}, batch.observations, actions)
        actor_loss = -(batch.discounts * q1).mean()
        return actor_loss
    
    # Compute gradient
    grads = jax.grad(actor_loss_fn)(actor_params)
    
    # Flatten gradient
    flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
    return flat_grads


def compute_corrected_gradient(
    state: TD3StateGC,
    batch: Batch
) -> jnp.ndarray:
    """
    Compute TD3-GC corrected actor gradient.
    
    Args:
        state: TD3StateGC state
        batch: Batch of data
        
    Returns:
        Flattened corrected gradient
    """
    # First compute standard gradient
    loss_fn = td3_loss_fn(state, batch)
    grads, _ = jax.grad(loss_fn, has_aux=True)(state.actor.params)
    
    # Apply gamma correction
    corrected_grads, _ = apply_gamma_correction(grads, state, batch)
    
    # Flatten corrected gradient
    flat_corrected_grads, _ = jax.flatten_util.ravel_pytree(corrected_grads)
    return flat_corrected_grads


def compute_true_gradient(
    state: TD3StateGC,
    true_critic: train_state.TrainState,
    batch: Batch
) -> jnp.ndarray:
    """
    Compute actor gradient using the true critic trained on fresh rollouts.
    
    Args:
        state: TD3StateGC state (for actor)
        true_critic: Freshly trained critic
        batch: Batch of data (from fresh rollouts)
        
    Returns:
        Flattened true gradient
    """
    # Define actor loss function using true critic
    def actor_loss_fn(actor_params: Params) -> jnp.ndarray:
        actions = state.actor.apply_fn({'params': actor_params}, batch.observations)
        q1, _ = true_critic.apply_fn({'params': true_critic.params}, batch.observations, actions)
        actor_loss = -(batch.discounts * q1).mean()
        return actor_loss
    
    # Compute gradient
    grads = jax.grad(actor_loss_fn)(state.actor.params)
    
    # Flatten gradient
    flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
    return flat_grads


def compute_cosine_similarity(grad1: jnp.ndarray, grad2: jnp.ndarray) -> float:
    """
    Compute cosine similarity between two flattened gradients.
    
    Args:
        grad1, grad2: Flattened gradient vectors
        
    Returns:
        Cosine similarity (-1 to 1, where 1 means perfect alignment)
    """
    # Compute norms
    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)
    
    # Handle zero gradients
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0  # No similarity for zero gradients
    
    # Compute cosine similarity
    dot_product = jnp.dot(grad1, grad2)
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return float(cosine_similarity)


def evaluate_gradient_quality(
    agent: TD3AgentGC,
    env: gym.Env,
    replay_buffer: ReplayBuffer,
    step_num: int,
    batch_size: int = 256,
    rollout_steps: int = 50_000,
    critic_training_steps: int = 5_000,
    evaluation_batch_size: int = 5000,
    num_parallel_envs: int = 10
) -> Dict[str, float]:
    """
    Main function to evaluate gradient quality by comparing with true gradient.
    Now uses copy.deepcopy and existing td3.py infrastructure.
    """
    print(f"\n=== Gradient Quality Evaluation at Step {step_num} ===")
    
    # 1. Collect fresh rollouts using current policy
    print("Collecting fresh rollouts...")
    #WARNING: Deterministic is False for but it should be TRUE
    rollout_obs, rollout_acts, rollout_rews, rollout_next_obs, rollout_masks, rollout_discounts = collect_rollouts(
        agent, env, num_steps=rollout_steps, deterministic=False, num_parallel_envs=num_parallel_envs, gamma=0.99
    )
    
    # 2. Train fresh critic on rollout data using existing infrastructure
    print("Training fresh critic...")
    true_critic = train_fresh_critic(
        rollout_obs, rollout_acts, rollout_rews, rollout_next_obs, rollout_masks,
        agent.state,  # Pass the full agent state
        seed=step_num, num_training_steps=critic_training_steps
    )
    
    # 3. Create batches for gradient comparison
    # Off-policy batch from replay buffer
    off_policy_batch = replay_buffer.sample(evaluation_batch_size)
    
    # True gradient batch from fresh rollouts (sample random subset)
    rollout_size = rollout_obs.shape[0]
    eval_indices = np.random.choice(rollout_size, size=evaluation_batch_size, replace=False)
    true_batch = Batch(
        observations=jax.device_put(rollout_obs[eval_indices]),
        actions=jax.device_put(rollout_acts[eval_indices]),
        rewards=jax.device_put(rollout_rews[eval_indices]),
        next_observations=jax.device_put(rollout_next_obs[eval_indices]),
        masks=jax.device_put(rollout_masks[eval_indices]),
        discounts=jax.device_put(rollout_discounts[eval_indices])  # Not used in actor loss
    )
    
    # 4. Compute gradients
    print("Computing gradients...")
    
    # Off-policy gradient (using replay buffer data and current critic)
    off_policy_grad = compute_actor_gradient(agent.state, off_policy_batch)
    
    # Corrected gradient (TD3-GC correction on replay buffer data)
    corrected_grad = compute_corrected_gradient(agent.state, off_policy_batch)
    

    
    # True gradient (using fresh rollout data and fresh critic)
    true_grad = compute_true_gradient(agent.state, true_critic, true_batch)
    
    # True approximation gradient (using fresh rollout data but current critic)
    true_approx_grad = compute_true_gradient(agent.state, agent.state.critic, true_batch)
    
    # 5. Compute cosine similarities
    print("Computing cosine similarities...")
    
    cosine_off_policy_vs_true = compute_cosine_similarity(off_policy_grad, true_grad)
    cosine_corrected_vs_true = compute_cosine_similarity(corrected_grad, true_grad)
    cosine_off_policy_vs_corrected = compute_cosine_similarity(off_policy_grad, corrected_grad)
    cosine_true_approx_vs_true = compute_cosine_similarity(true_approx_grad, true_grad)
    
    # 6. Compute gradient norms for additional insight
    norm_off_policy = float(jnp.linalg.norm(off_policy_grad))
    norm_corrected = float(jnp.linalg.norm(corrected_grad))
    norm_true = float(jnp.linalg.norm(true_grad))
    norm_true_approx = float(jnp.linalg.norm(true_approx_grad))
    
    # 7. Create results dictionary
    results = {
        'cosine_similarity/off_policy_vs_true': cosine_off_policy_vs_true,
        'cosine_similarity/corrected_vs_true': cosine_corrected_vs_true,
        'cosine_similarity/off_policy_vs_corrected': cosine_off_policy_vs_corrected,
        'cosine_similarity/true_approx_vs_true': cosine_true_approx_vs_true,
        'gradient_improvement': cosine_corrected_vs_true - cosine_off_policy_vs_true,  # Positive means correction helps
        'step': step_num
    }
    
    # 8. Log results
    print(f"Off-policy vs True Cosine Similarity: {cosine_off_policy_vs_true:.4f}")
    print(f"Corrected vs True Cosine Similarity: {cosine_corrected_vs_true:.4f}")
    print(f"Off-policy vs Corrected Cosine Similarity: {cosine_off_policy_vs_corrected:.4f}")
    print(f"True Approx vs True Cosine Similarity: {cosine_true_approx_vs_true:.4f}")
    print(f"Gradient Improvement: {results['gradient_improvement']:.4f}")
    print(f"Gradient Norms - Off-policy: {norm_off_policy:.4f}, Corrected: {norm_corrected:.4f}, True: {norm_true:.4f}, True Approx: {norm_true_approx:.4f}")
    
    # Interpret results
    print("\nResult Interpretation:")
    print("-" * 40)
    if results['gradient_improvement'] > 0:
        print(f"✅ Gradient correction is helping! Improvement: {results['gradient_improvement']:.6f}")
    else:
        print(f"❌ Gradient correction may not be helping. Change: {results['gradient_improvement']:.6f}")
    
    print(f"Off-policy gradient has {cosine_off_policy_vs_true:.6f} cosine similarity with true gradient")
    print(f"Corrected gradient has {cosine_corrected_vs_true:.6f} cosine similarity with true gradient")
    print(f"True approx gradient has {cosine_true_approx_vs_true:.6f} cosine similarity with true gradient")
    
    # Log to wandb if available
    try:
        wandb.log(results, step=step_num)
        print("Results logged to Wandb.")
    except Exception as e:
        print(f"Failed to log to Wandb: {e}")
    
    print("=== Gradient Quality Evaluation Complete ===\n")
    
    return results


