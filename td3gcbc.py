# td3gcbc.py - TD3 with Gradient Correction and Behavior Cloning
import os
import time
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import distrax
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import minari
import numpy as np
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from flax import struct

# Import TD3-GC components (includes gamma critic and correction)
from td3_gc import (
    TD3StateGC,
    TD3ConfigGC, 
    update_gamma_critic,
    target_update,
    create_td3_gc_learner,
)

# Import TD3-BC components for BC loss and dataset handling
from td3bc import (
    get_dataset,
    evaluate,
    Transition,
    RunTimeConfig
)

# Import TD3 base components
from td3 import (
    update_critic,
    _td3_sample_eval_step
)

# Import composable actor updates
from actor_updates import update_td3gcbc_actor

from utils import Batch, MLP, default_init, PRNGKey, Params, InfoDict
from networks import DoubleCritic, DeterministicActor

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

ACTOR_INFO_TEMPLATE = {
    'actor_loss': jnp.nan,
    'td3_loss': jnp.nan,
    'bc_loss': jnp.nan,
    'loss_lambda': jnp.nan,
    'original_grad_norm': jnp.nan,
    'corrected_grad_norm': jnp.nan,
    'grad_cosine_similarity': jnp.nan,
    'grad_cosine_distance': jnp.nan,
}

@struct.dataclass(kw_only=True)
class TD3GCBCConfig(TD3ConfigGC):
    """Extends TD3ConfigGC with TD3-BC specific parameters."""
    # TD3-BC specific parameters (no defaults)
    alpha: float              # BC loss weight
    use_bc_loss: bool         # Enable behavior cloning component
    n_jitted_updates: int     # Number of updates per training step
    batch_size: int
    max_steps: int
    
    # Dataset parameters
    data_size: int
    normalize_state: bool
    
    def __post_init__(self):
        # Ensure hidden_dims is a tuple for JAX hashability
        if hasattr(self, "hidden_dims") and not isinstance(self.hidden_dims, tuple):
            object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))




# --- Batch Update Function ---
def update_n_times(
    state: TD3StateGC, 
    dataset: Transition,
    config: TD3GCBCConfig,
) -> Tuple[TD3StateGC, Dict]:
    """Optimized version using Python conditionals like TD3BC."""
    
    critic_loss = actor_loss = gamma_loss = 0.0
    
    for step in range(config.n_jitted_updates):
        # Sample batch (same as before)
        rng, batch_rng = jax.random.split(state.rng, 2)
        batch_idx = jax.random.randint(
            batch_rng, (config.batch_size,), 0, len(dataset.observations)
        )
        batch_transition = jax.tree_util.tree_map(lambda x: x[batch_idx], dataset)
        batch = Batch(
            observations=batch_transition.observations,
            actions=batch_transition.actions,
            rewards=batch_transition.rewards,
            next_observations=batch_transition.next_observations,
            discounts=jnp.ones_like(batch_transition.rewards),
            masks=1.0 - batch_transition.dones
        )
        
        rng, critic_rng, gamma_rng = jax.random.split(state.rng, 3)
        
        # 1. Update critic
        new_critic, critic_info = update_critic(critic_rng, state, batch)
        state = state.replace(critic=new_critic)
        critic_loss = critic_info['critic_loss']
        
        # 2. Update gamma critic
        new_gamma_critic, gamma_info = update_gamma_critic(gamma_rng, state, batch)
        state = state.replace(gamma_critic=new_gamma_critic)
        gamma_loss = gamma_info['gamma_loss']
        
        # 3. Conditional target gamma update
        if step % config.target_gamma_critic_update_period == 0:
            new_target_gamma_params = target_update(
                new_gamma_critic.params, state.target_gamma_critic_params, config.tau
            )
            state = state.replace(target_gamma_critic_params=new_target_gamma_params)
        
        # 4. Conditional actor and target updates
        if step % config.policy_delay == 0:
            new_actor, actor_info = update_td3gcbc_actor(state, batch)
            
            new_target_critic_params = target_update(
                new_critic.params, state.target_critic_params, config.tau
            )
            new_target_actor_params = target_update(
                new_actor.params, state.target_actor_params, config.tau
            )
            
            state = state.replace(
                actor=new_actor,
                target_critic_params=new_target_critic_params,
                target_actor_params=new_target_actor_params
            )
            actor_loss = actor_info['actor_loss']
    
    # Update step counter and RNG
    state = state.replace(rng=rng, step=state.step + config.n_jitted_updates)
    
    return state, {
        "critic_loss": critic_loss,
        "gamma_loss": gamma_loss,
        "actor_loss": actor_loss,
    }


# --- Factory Function ---
def create_td3gcbc_train_state(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: TD3GCBCConfig,
) -> TD3StateGC:
    """Creates TD3-GCBC state by reusing TD3-GC factory function."""
    return create_td3_gc_learner(seed, observations, actions, config)


# --- TD3-GCBC Class for compatibility ---
class TD3GCBC(object):
    """Wrapper class for TD3-GCBC implementation."""
    
    @classmethod
    def update_n_times(
        cls,
        train_state: TD3StateGC,
        data: Transition,
        config: TD3GCBCConfig,
    ) -> Tuple[TD3StateGC, Dict]:
        return update_n_times(train_state, data, config)

    @classmethod 
    def get_action(
        cls,
        train_state: TD3StateGC,
        obs: jnp.ndarray,
        max_action: float = 1.0,
    ) -> jnp.ndarray:
        """Get deterministic action for evaluation."""
        action = _td3_sample_eval_step(train_state, obs)
        return action.clip(-max_action, max_action)


if __name__ == "__main__":
    # Load base config from file, then merge with CLI overrides
    base_config = OmegaConf.load("configs/algorithm/td3gcbc.yaml")
    cli_config = OmegaConf.from_cli()
    
    # Merge: CLI overrides base config
    conf_dict = OmegaConf.merge(base_config, cli_config)
    
    # Convert OmegaConf to regular Python dict - THIS IS THE KEY FIX
    conf_dict = OmegaConf.to_container(conf_dict, resolve=True)
    
    config = TD3GCBCConfig(**conf_dict)
    runtime_config = RunTimeConfig()
    # runtime_config.agent_name = "TD3-GCBC"
    # runtime_config.project = "train-TD3-GCBC"
    
    wandb.init(project=runtime_config.project, config=config)
    
    # Load dataset and environment
    minari_dataset = minari.load_dataset(runtime_config.dataset_name, download=True)
    env = minari_dataset.recover_environment()
    
    dataset, obs_mean, obs_std = get_dataset(minari_dataset, config)
    
    # Create train state
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = create_td3gcbc_train_state(
        runtime_config.seed, example_batch.observations, example_batch.actions, config
    )
    
    # JIT functions
    algo = TD3GCBC()
    update_fn = jax.jit(update_n_times, static_argnums=(2,))
    act_fn = jax.jit(algo.get_action)

    # Training loop
    num_steps = config.max_steps // config.n_jitted_updates
    eval_interval = runtime_config.eval_interval // config.n_jitted_updates
    
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        train_state, update_info = update_fn(train_state, dataset, config)
        
        if i % runtime_config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % eval_interval == 0:
            policy_fn = partial(act_fn, train_state=train_state)
            normalized_score = evaluate(
                policy_fn, env, num_episodes=runtime_config.eval_episodes,
                obs_mean=obs_mean, obs_std=obs_std, minari_dataset=minari_dataset
            )
            print(i, normalized_score)
            eval_metrics = {f"{runtime_config.dataset_name}/normalized_score": normalized_score}
            wandb.log(eval_metrics, step=i)
    
    # Final evaluation
    policy_fn = partial(act_fn, train_state=train_state)
    normalized_score = evaluate(
        policy_fn, env, num_episodes=runtime_config.eval_episodes,
        obs_mean=obs_mean, obs_std=obs_std, minari_dataset=minari_dataset
    )
    print("Final Evaluation Score:", normalized_score)
    wandb.log({f"{runtime_config.dataset_name}/final_normalized_score": normalized_score})
    wandb.finish()
