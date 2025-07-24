import minari

dataset = minari.load_dataset('mujoco/hopper/expert-v0',download=True)
env  = dataset.recover_environment()
eval_env = dataset.recover_environment(eval_env=True)

assert env.spec == eval_env.spec


# source https://github.com/sfujim/TD3_BC
# https://arxiv.org/abs/2106.06860
# Modified to use Minari datasets instead of D4RL and to maximally reuse TD3 components
# Usage example:
#   python td3bc.py dataset_name=D4RL/hopper/medium-expert-v2
import os
from functools import partial
from typing import Callable, Dict, NamedTuple, Sequence, Tuple


import gymnasium as gym
import jax
import jax.numpy as jnp
import minari
import numpy as np
import tqdm
import wandb
from omegaconf import OmegaConf
from flax import struct
from flax.training.train_state import TrainState

# Import reusable components from TD3
from td3 import (
    update_critic,           # Identical critic update logic
    target_update,          # Identical target network update
    TD3State,               # Base state structure
    TD3Config,              # Base configuration
    create_td3_learner,     # Factory function for initialization
    _td3_sample_eval_step   # Evaluation sampling
)
from utils import Batch, InfoDict
# Import composable actor updates
from actor_updates import update_td3bc_actor

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@struct.dataclass(kw_only=True)
class TD3BCConfig(TD3Config):
    """Extends TD3Config with TD3BC-specific parameters."""
    # Provide defaults for all TD3Config fields to fix dataclass inheritance
    agent_name: str = struct.field(pytree_node=False, default="TD3-BC")
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_dims: Sequence[int] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.0  # No exploration in offline setting
    max_action: float = 1.0
    final_fc_init_scale: float = 1e-2
    use_layer_norm: bool = True
    

    batch_size: int = 256
    max_steps: int = int(1e6)
    alpha: float = 2.5              # BC loss weight
    use_bc_loss: bool = True        # Enable behavior cloning component
    n_jitted_updates: int = 8       # Number of updates per training step
    
    # Dataset parameters
    data_size: int = int(1e6)
    normalize_state: bool = True


@struct.dataclass
class RunTimeConfig():
    # TD3BC-specific parameters with defaults
    algo: str = "TD3-BC"
    project: str = "train-TD3-BC"
    agent_name: str = "TD3-BC"
    dataset_name: str = 'mujoco/hopper/expert-v0'
    seed: int = 42
    eval_episodes: int = 5
    log_interval: int = 100000
    eval_interval: int = 40_000



conf_dict = OmegaConf.from_cli()
config = TD3BCConfig(**conf_dict)
runtime_config = RunTimeConfig()


@struct.dataclass
class TD3BCState(TD3State):
    """Extends TD3State with TD3BC-specific configuration."""
    config: TD3BCConfig


# Transition class for dataset handling (kept for compatibility)
class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray







def get_dataset(
    minari_dataset, config: TD3BCConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
    # Convert Minari dataset to the expected format
    observations_list = []
    actions_list = []
    rewards_list = []
    next_observations_list = []
    dones_list = []
    
    for episode_data in minari_dataset.iterate_episodes():
        episode_length = len(episode_data.observations)
        observations_list.extend(episode_data.observations[:-1])  # exclude last obs
        actions_list.extend(episode_data.actions)
        rewards_list.extend(episode_data.rewards)
        next_observations_list.extend(episode_data.observations[1:])  # exclude first obs
        
        # Create dones array: False for all steps except the last one
        episode_dones = [False] * (episode_length - 1)
        if len(episode_dones) > 0:
            episode_dones[-1] = episode_data.terminations[-1] or episode_data.truncations[-1]
        dones_list.extend(episode_dones)
    
    # Convert to numpy arrays
    dataset = {
        "observations": np.array(observations_list, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.float32),
        "rewards": np.array(rewards_list, dtype=np.float32),
        "next_observations": np.array(next_observations_list, dtype=np.float32)
    }

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones = np.array(dones_list, dtype=np.float32)

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
    )
    # shuffle data and select the first data_size samples
    data_size = min(config.data_size, len(dataset.observations))
    rng = jax.random.PRNGKey(runtime_config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    # normalize states
    obs_mean, obs_std = 0, 1
    if config.normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    return dataset, obs_mean, obs_std


# --- TD3BC Actor Update (extends TD3 with BC loss) ---
def update_actor_bc(state: TD3BCState, batch: Batch) -> Tuple[TrainState, InfoDict]:
    """TD3 actor update extended with behavior cloning loss using composable system."""
    return update_td3bc_actor(state, batch)


# --- Batch Update Function (TD3BC's key feature) ---

def update_n_times(
    state: TD3BCState, 
    dataset: Transition,  # Full dataset 
    config: TD3BCConfig,
) -> Tuple[TD3BCState, Dict]:
    """Performs n_jitted_updates in a single JIT-compiled function."""
    
    
    critic_loss = actor_loss = 0.0  # Initialize for return
    
    for step in range(config.n_jitted_updates):
        # Sample random batch from dataset
        rng, batch_rng = jax.random.split(state.rng, 2)
        batch_idx = jax.random.randint(
            batch_rng, (config.batch_size,), 0, len(dataset.observations)
        )
        batch_transition = jax.tree_util.tree_map(lambda x: x[batch_idx], dataset)
        
        # Convert Transition to Batch format for TD3 functions
        batch = Batch(
            observations=batch_transition.observations,
            actions=batch_transition.actions,
            rewards=batch_transition.rewards,
            next_observations=batch_transition.next_observations,
            discounts=jnp.ones_like(batch_transition.rewards),  # Not used in offline setting
            masks=1.0 - batch_transition.dones  # Convert dones to masks
        )
        
        rng, critic_rng = jax.random.split(state.rng, 2)
        
        # 1. Update critic (reuse TD3 function directly!)
        new_critic, critic_info = update_critic(critic_rng, state, batch)
        state = state.replace(critic=new_critic)
        critic_loss = critic_info['critic_loss']  # Store for return
        
        # 2. Conditional actor update (with TD3's policy delay)
        if step % config.policy_delay == 0:
            # Update actor with BC loss
            new_actor, actor_info = update_actor_bc(state, batch)
            
            # Update target networks (reuse TD3 function!)
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
            actor_loss = actor_info['actor_loss']  # Store for return
    
    # Update step counter and RNG
    state = state.replace(rng=rng, step=state.step + config.n_jitted_updates)
    
    return state, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }


# --- Factory Function (reuses TD3 factory with config conversion) ---
def create_td3bc_train_state(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: TD3BCConfig,
) -> TD3BCState:
    """Creates TD3BC state by reusing TD3 factory function."""
    
    # Create base TD3 state using existing factory (pass seed, not rng)
    base_state = create_td3_learner(seed, observations, actions, config)
    
    # Convert to TD3BC state (just changes the type and config)
    return TD3BCState(
        actor=base_state.actor,
        critic=base_state.critic,
        target_actor_params=base_state.target_actor_params,
        target_critic_params=base_state.target_critic_params,
        rng=base_state.rng,
        step=base_state.step,
        config=config  # Use TD3BC config
    )


def evaluate(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env: gym.Env,
    num_episodes: int,
    obs_mean,
    obs_std,
    minari_dataset=None,
) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            observation = (observation - obs_mean) / obs_std
            action = policy_fn(obs=observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    
    mean_return = np.mean(episode_returns)
    # Use Minari's normalized score if available, otherwise return raw score
    #return minari.get_normalized_score(dataset=minari_dataset,returns=np.array(episode_returns))
    return mean_return


# --- TD3BC Class for backward compatibility ---
class TD3BC(object):
    """Wrapper class that provides the same interface as the original TD3BC implementation."""
    
    @classmethod
    def update_n_times(
        cls,
        train_state: TD3BCState,
        data: Transition,
    ) -> Tuple[TD3BCState, Dict]:
        """Calls the refactored update_n_times function."""
        return update_n_times(train_state, data, config)

    @classmethod 
    def get_action(
        cls,
        train_state: TD3BCState,
        obs: jnp.ndarray,
        max_action: float = 1.0,
    ) -> jnp.ndarray:
        """Get deterministic action for evaluation (reuses TD3 logic)."""
        action = _td3_sample_eval_step(train_state, obs)
        return action.clip(-max_action, max_action)


if __name__ == "__main__":
    wandb.init(project=runtime_config.project, config=config)
    
    # Load Minari dataset and environment
    minari_dataset = minari.load_dataset(runtime_config.dataset_name, download=True)
    env = minari_dataset.recover_environment()
    
    dataset, obs_mean, obs_std = get_dataset(minari_dataset, config)
    
    # Create train_state using refactored factory
    rng = jax.random.PRNGKey(runtime_config.seed)
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = create_td3bc_train_state(
        runtime_config.seed, example_batch.observations, example_batch.actions, config
    )
    
    # Use wrapper class for compatibility
    algo = TD3BC()
    update_fn = jax.jit(update_n_times, static_argnums=(2,))  # JIT the refactored function directly (no static args needed)
    act_fn = jax.jit(algo.get_action)

    num_steps = config.max_steps // config.n_jitted_updates
    eval_interval = runtime_config.eval_interval // config.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        train_state, update_info = update_fn(
            train_state,
            dataset,
            config
        )  # update parameters
        if i % runtime_config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % eval_interval == 0:
            policy_fn = partial(act_fn, train_state=train_state)
            normalized_score = evaluate(
                policy_fn,
                env,
                num_episodes=runtime_config.eval_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
                minari_dataset=minari_dataset,
            )
            print(i, normalized_score)
            eval_metrics = {f"{runtime_config.dataset_name}/normalized_score": normalized_score}
            wandb.log(eval_metrics, step=i)
    # final evaluation
    policy_fn = partial(act_fn, train_state=train_state)
    normalized_score = evaluate(
        policy_fn,
        env,
        num_episodes=runtime_config.eval_episodes,
        obs_mean=obs_mean,
        obs_std=obs_std,
        minari_dataset=minari_dataset,
    )
    print("Final Evaluation Score:", normalized_score)
    wandb.log({f"{runtime_config.dataset_name}/final_normalized_score": normalized_score})
    wandb.finish()
