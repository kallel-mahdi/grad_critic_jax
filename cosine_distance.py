# cosine_distance.py
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import wandb
from tqdm import tqdm

from td3_gc import TD3AgentGC, TD3StateGC, create_td3_gc_learner, TD3ConfigGC
from td3 import update_critic as base_update_critic
from networks import DoubleCritic
from utils import Batch, ReplayBuffer, PRNGKey, Params, InfoDict
from actor_updates import td3_loss_fn, apply_gamma_correction
import flax.linen as nn
from flax.training import train_state
import optax
import jax.flatten_util


def collect_rollouts(
    agent: TD3AgentGC,
    env: gym.Env,
    num_steps: int = 100_000,
    deterministic: bool = True,
    num_parallel_envs: int = 8
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
    
    vec_env = gym.vector.make(env_id, num_envs=num_parallel_envs, asynchronous=True)
    print(f"Created vectorized environment with {num_parallel_envs} parallel environments")

    

    observations = []
    actions = []
    rewards = []
    next_observations = []
    masks = []
    
    # Reset all environments
    obs, _ = vec_env.reset()
    steps_collected = 0
    
    # Calculate steps per environment
    steps_per_env = num_steps // num_parallel_envs
    total_steps_needed = steps_per_env * num_parallel_envs
    
    progress_bar = tqdm(total=total_steps_needed, desc="Collecting parallel rollouts")
    
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
        
        print(f"Collected {observations.shape[0]} transitions using {num_parallel_envs} parallel environments")
        
        return (
            observations.astype(np.float32),
            actions.astype(np.float32),
            rewards.astype(np.float32),
            next_observations.astype(np.float32),
            masks.astype(np.float32)
        )
    

def train_fresh_critic(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    masks: jnp.ndarray,
    actor_params: Params,
    actor_apply_fn: Any,
    config: TD3ConfigGC,
    seed: int = 42,
    num_training_steps: int = 50_000
) -> train_state.TrainState:
    """
    Train a fresh critic from scratch on rollout data.
    
    Args:
        observations, actions, rewards, next_observations, masks: Rollout data
        actor_params: Current actor parameters for target Q computation
        actor_apply_fn: Actor apply function
        config: TD3-GC configuration
        seed: Random seed for critic initialization
        num_training_steps: Number of training steps for the critic
        
    Returns:
        Trained critic TrainState
    """
    rng = jax.random.PRNGKey(seed)
    rng, critic_key = jax.random.split(rng)
    
    # Initialize fresh critic with same architecture as current critic
    critic_def = DoubleCritic(config.hidden_dims, use_layer_norm=config.use_layer_norm)
    sample_obs = jnp.expand_dims(observations[0], 0)
    sample_act = jnp.expand_dims(actions[0], 0)
    critic_params = critic_def.init(critic_key, sample_obs, sample_act)['params']
    
    critic = train_state.TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=config.critic_lr)
    )
    
    # Convert data to JAX arrays
    observations = jax.device_put(observations)
    actions = jax.device_put(actions)
    rewards = jax.device_put(rewards)
    next_observations = jax.device_put(next_observations)
    masks = jax.device_put(masks)
    
    # Training loop for critic
    dataset_size = observations.shape[0]
    batch_size = min(256, dataset_size // 10)  # Use reasonable batch size
    
    for step in tqdm(range(num_training_steps), desc="Training fresh critic"):
        rng, batch_rng, noise_rng = jax.random.split(rng, 3)
        
        # Sample batch
        batch_idx = jax.random.randint(batch_rng, (batch_size,), 0, dataset_size)
        batch_obs = observations[batch_idx]
        batch_act = actions[batch_idx]
        batch_rew = rewards[batch_idx]
        batch_next_obs = next_observations[batch_idx]
        batch_masks = masks[batch_idx]
        
        # Compute target using current actor (deterministic)
        next_actions = actor_apply_fn({'params': actor_params}, batch_next_obs)
        
        # Add target policy smoothing noise (like TD3)
        noise = (jax.random.normal(noise_rng, next_actions.shape) * config.policy_noise
                ).clip(-config.noise_clip, config.noise_clip)
        next_actions = (next_actions + noise).clip(-config.max_action, config.max_action)
        
        # Compute target Q-values using current critic
        next_q1, next_q2 = critic.apply_fn({'params': critic.params}, batch_next_obs, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        
        # Bellman target
        target_q = batch_rew + config.discount * batch_masks * next_q
        target_q = jax.lax.stop_gradient(target_q)
        
        # Critic loss function
        def critic_loss_fn(critic_params: Params) -> jnp.ndarray:
            q1, q2 = critic.apply_fn({'params': critic_params}, batch_obs, batch_act)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            return critic_loss
        
        # Update critic
        grads = jax.grad(critic_loss_fn)(critic.params)
        critic = critic.apply_gradients(grads=grads)
    
    return critic


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


def compute_cosine_distance(grad1: jnp.ndarray, grad2: jnp.ndarray) -> float:
    """
    Compute cosine distance between two flattened gradients.
    
    Args:
        grad1, grad2: Flattened gradient vectors
        
    Returns:
        Cosine distance (1 - cosine similarity)
    """
    # Compute norms
    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)
    
    # Handle zero gradients
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 1.0  # Maximum distance for zero gradients
    
    # Compute cosine similarity
    dot_product = jnp.dot(grad1, grad2)
    cosine_similarity = dot_product / (norm1 * norm2)
    
    # Compute cosine distance
    cosine_distance = 1.0 - cosine_similarity
    
    return float(cosine_distance)


def evaluate_gradient_quality(
    agent: TD3AgentGC,
    env: gym.Env,
    replay_buffer: ReplayBuffer,
    step_num: int,
    batch_size: int = 256,
    rollout_steps: int = 50_000,
    critic_training_steps: int = 50_000,
    evaluation_batch_size: int = 1000,
    num_parallel_envs: int = 8
) -> Dict[str, float]:
    """
    Main function to evaluate gradient quality by comparing with true gradient.
    
    Args:
        agent: Current TD3-GC agent
        env: Environment for rollouts (used to create vectorized environments)
        replay_buffer: Current replay buffer for off-policy data
        step_num: Current training step number
        batch_size: Batch size for gradient computation
        rollout_steps: Number of steps to rollout for true gradient
        critic_training_steps: Number of steps to train fresh critic
        evaluation_batch_size: Batch size for final gradient comparison
        num_parallel_envs: Number of parallel environments for faster rollouts
        
    Returns:
        Dictionary with cosine distance metrics
    """
    print(f"\n=== Gradient Quality Evaluation at Step {step_num} ===")
    
    # 1. Collect fresh rollouts using current policy
    print("Collecting fresh rollouts...")
    rollout_obs, rollout_acts, rollout_rews, rollout_next_obs, rollout_masks = collect_rollouts(
        agent, env, num_steps=rollout_steps, deterministic=True, num_parallel_envs=num_parallel_envs
    )
    
    # 2. Train fresh critic on rollout data
    print("Training fresh critic...")
    true_critic = train_fresh_critic(
        rollout_obs, rollout_acts, rollout_rews, rollout_next_obs, rollout_masks,
        agent.state.actor.params, agent.state.actor.apply_fn, agent.state.config,
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
        discounts=jnp.ones(evaluation_batch_size)  # Not used in actor loss
    )
    
    # 4. Compute gradients
    print("Computing gradients...")
    
    # Off-policy gradient (using replay buffer data and current critic)
    off_policy_grad = compute_actor_gradient(agent.state, off_policy_batch)
    
    # Corrected gradient (TD3-GC correction on replay buffer data)
    corrected_grad = compute_corrected_gradient(agent.state, off_policy_batch)
    
    # True gradient (using fresh rollout data and fresh critic)
    true_grad = compute_true_gradient(agent.state, true_critic, true_batch)
    
    # 5. Compute cosine distances
    print("Computing cosine distances...")
    
    cosine_off_policy_vs_true = compute_cosine_distance(off_policy_grad, true_grad)
    cosine_corrected_vs_true = compute_cosine_distance(corrected_grad, true_grad)
    cosine_off_policy_vs_corrected = compute_cosine_distance(off_policy_grad, corrected_grad)
    
    # 6. Compute gradient norms for additional insight
    norm_off_policy = float(jnp.linalg.norm(off_policy_grad))
    norm_corrected = float(jnp.linalg.norm(corrected_grad))
    norm_true = float(jnp.linalg.norm(true_grad))
    
    # 7. Create results dictionary
    results = {
        'cosine_distance/off_policy_vs_true': cosine_off_policy_vs_true,
        'cosine_distance/corrected_vs_true': cosine_corrected_vs_true,
        'cosine_distance/off_policy_vs_corrected': cosine_off_policy_vs_corrected,
        'gradient_norms/off_policy': norm_off_policy,
        'gradient_norms/corrected': norm_corrected,
        'gradient_norms/true': norm_true,
        'gradient_improvement': cosine_off_policy_vs_true - cosine_corrected_vs_true,  # Positive means correction helps
        'step': step_num
    }
    
    # 8. Log results
    print(f"Off-policy vs True Cosine Distance: {cosine_off_policy_vs_true:.4f}")
    print(f"Corrected vs True Cosine Distance: {cosine_corrected_vs_true:.4f}")
    print(f"Off-policy vs Corrected Cosine Distance: {cosine_off_policy_vs_corrected:.4f}")
    print(f"Gradient Improvement: {results['gradient_improvement']:.4f}")
    print(f"Gradient Norms - Off-policy: {norm_off_policy:.4f}, Corrected: {norm_corrected:.4f}, True: {norm_true:.4f}")
    
    # Log to wandb if available
    try:
        wandb.log(results, step=step_num)
        print("Results logged to Wandb.")
    except Exception as e:
        print(f"Failed to log to Wandb: {e}")
    
    print("=== Gradient Quality Evaluation Complete ===\n")
    
    return results


# Convenience function to integrate into main training loop
def maybe_evaluate_gradient_quality(
    agent: TD3AgentGC,
    env: gym.Env,
    replay_buffer: ReplayBuffer,
    step_num: int,
    evaluation_frequency: int = 50_000,
    num_parallel_envs: int = 8,
    **kwargs
) -> Dict[str, float]:
    """
    Conditionally evaluate gradient quality every evaluation_frequency steps.
    
    Args:
        agent: Current TD3-GC agent
        env: Environment for rollouts (used to create vectorized environments)
        replay_buffer: Current replay buffer
        step_num: Current training step
        evaluation_frequency: How often to evaluate (default: 50K steps)
        num_parallel_envs: Number of parallel environments for faster rollouts
        **kwargs: Additional arguments passed to evaluate_gradient_quality
        
    Returns:
        Dictionary with results if evaluation performed, empty dict otherwise
    """
    if step_num % evaluation_frequency == 0 and step_num > 0:
        return evaluate_gradient_quality(
            agent, env, replay_buffer, step_num, num_parallel_envs=num_parallel_envs, **kwargs
        )
    else:
        return {}
