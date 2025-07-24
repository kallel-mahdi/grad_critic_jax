# td3gcbc.py - TD3 with Gradient Correction and Behavior Cloning
import os
from functools import partial
from typing import Dict, Tuple


import jax
import jax.numpy as jnp
import minari

import tqdm
import wandb
from omegaconf import OmegaConf
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

from utils import Batch, InfoDict  

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
    use_layer_norm: bool 
    
    # Dataset parameters
    data_size: int
    normalize_state: bool
    
    def __post_init__(self):
        # Ensure hidden_dims is a tuple for JAX hashability
        if hasattr(self, "hidden_dims") and not isinstance(self.hidden_dims, tuple):
            object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))


# --- Single TD3-GCBC Update Step ---
@partial(jax.jit, static_argnums=(2))
def _td3_gcbc_update_step(state: TD3StateGC, dataset: Transition, batch_size: int):
    """Single TD3-GCBC update step (external JIT)."""
    rng, key_critic, key_gamma = jax.random.split(state.rng, 3)

    # Sample batch INSIDE JIT
    rng, batch_rng = jax.random.split(state.rng, 2)
    batch_idx = jax.random.randint(batch_rng, (batch_size,), 0, len(dataset.observations))
    batch_transition = jax.tree_util.tree_map(lambda x: x[batch_idx], dataset)

    # 1. Update critic
    new_critic, critic_info = update_critic(key_critic, state, batch_transition)
    temp_state = state.replace(critic=new_critic)

    # 2. Update gamma critic
    new_gamma_critic, gamma_info = update_gamma_critic(key_gamma, temp_state, batch_transition)
    temp_state = temp_state.replace(gamma_critic=new_gamma_critic)

    # 3. Conditional target gamma update
    def _update_target_gamma(state):
        return target_update(state.gamma_critic.params, state.target_gamma_critic_params, state.config.tau)
    
    def _no_update_target_gamma(state):
        return state.target_gamma_critic_params

    new_target_gamma_params = jax.lax.cond(
        state.step % state.config.target_gamma_critic_update_period == 0,
        _update_target_gamma,
        _no_update_target_gamma,
        temp_state
    )

    # 4. Conditional actor and target updates
    def _update_actor_and_targets(state_for_actor_update):
        new_actor, actor_info = update_td3gcbc_actor(state_for_actor_update, batch_transition)
        
        new_target_critic_params = target_update(
            state_for_actor_update.critic.params, 
            state_for_actor_update.target_critic_params, 
            state_for_actor_update.config.tau
        )
        new_target_actor_params = target_update(
            new_actor.params, 
            state_for_actor_update.target_actor_params, 
            state_for_actor_update.config.tau
        )
        
        return new_actor, new_target_actor_params, new_target_critic_params, actor_info

    def _no_update_actor_and_targets(state_for_actor_update):
        return (state_for_actor_update.actor, 
                state_for_actor_update.target_actor_params, 
                state_for_actor_update.target_critic_params, 
                ACTOR_INFO_TEMPLATE)

    new_actor, new_target_actor_params, new_target_critic_params, actor_info = jax.lax.cond(
        state.step % state.config.policy_delay == 0,
        _update_actor_and_targets,
        _no_update_actor_and_targets,
        temp_state
    )

    # Create final new state
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

    return new_state, {**critic_info, **gamma_info, **actor_info}


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
    def update(
        cls,
        train_state: TD3StateGC,
        batch: Batch,
    ) -> Tuple[TD3StateGC, Dict]:
        """Single TD3-GCBC update step."""
        return _td3_gcbc_update_step(train_state, batch,config.batch_size)

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
    update_fn = jax.jit(algo.update)  # Single update instead of update_n_times
    act_fn = jax.jit(algo.get_action)

    # Training loop - now we handle the multiple updates here
    total_updates = 0
    max_updates = config.max_steps
    eval_interval = runtime_config.eval_interval
    
    for i in tqdm.tqdm(range(1, max_updates + 1), smoothing=0.1, dynamic_ncols=True):
        # Sample batch for this update
        rng, batch_rng = jax.random.split(train_state.rng, 2)
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
        
        train_state, update_info = update_fn(train_state, batch)
        
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
            print(eval_metrics)
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
