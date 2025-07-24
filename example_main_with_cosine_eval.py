# example_main_with_cosine_eval.py 
# Example showing how to integrate cosine distance evaluation into main.py

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from sac import SACAgent
from sac_gc import SACAgentGC
from td3 import TD3Agent
from td3_gc import TD3AgentGC
from crossq import CrossQAgent
from crossq_gc import CrossQAgentGC
from crossq3 import CrossQTD3Agent
from utils import ReplayBuffer, evaluate

# Import the cosine distance evaluation module
from cosine_distance import maybe_evaluate_gradient_quality

import wandb
import jax
import jax.numpy as jnp

os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    # --- Setup (same as before) ---
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.local_devices()}")

    x = jnp.array([1, 2, 3])
    print(f"Simple JAX operation result: {x + 1}")
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    # Initialize Weights & Biases
    if cfg.logging.use_wandb:
        run_name = f"{cfg.algorithm.agent_name}_{cfg.environment.name}_{cfg.seed}"
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            sync_tensorboard=False,
            monitor_gym=False,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=False,
                init_timeout=1800,
            ),
            save_code=True,
            dir=output_dir
        )

    # Create environment
    env = gym.make(cfg.environment.name)
    eval_env = gym.make(cfg.environment.name)
    
    # Create a separate environment for gradient quality evaluation rollouts
    # This ensures we don't interfere with the main training environment
    gradient_eval_env = gym.make(cfg.environment.name) if cfg.algorithm.agent_name == "TD3GC" else None
    
    # Set random seeds
    np.random.seed(cfg.seed)

    # Get observation and action space info
    obs_space = env.observation_space
    act_space = env.action_space
    sample_obs = np.zeros((1, obs_space.shape[0]), dtype=np.float32)
    sample_act = np.zeros((1, act_space.shape[0]), dtype=np.float32)
    
    agent_config = OmegaConf.to_container(cfg.algorithm, resolve=True)

    # --- Agent Initialization (same as before) ---
    agent_classes = {
        "SAC": SACAgent,
        "SACGC": SACAgentGC,
        "TD3": TD3Agent,
        "TD3GC": TD3AgentGC,
        "CrossQ": CrossQAgent,
        "CrossQGC": CrossQAgentGC,
        "CrossQ3": CrossQTD3Agent,
    }

    agent_name = cfg.algorithm.agent_name
    if agent_name not in agent_classes:
        raise ValueError(f"Unknown agent name: {agent_name}. Available: {list(agent_classes.keys())}")

    SelectedAgentClass = agent_classes[agent_name]
    print(f"Using agent: {agent_name}")

    agent = SelectedAgentClass.create(
        seed=cfg.seed,
        observations=sample_obs,
        actions=sample_act,
        agent_config=agent_config
    )

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        observation_space=obs_space,
        action_space=act_space,
        capacity=cfg.training.replay_buffer_capacity
    )

    # Training loop
    observation, _ = env.reset(seed=cfg.seed)
    discount = 1.
    
    for step_num in tqdm(range(1, cfg.training.max_steps + 1), desc=f"Training {cfg.environment.name}"):
        if step_num < cfg.training.start_steps:
            action = env.action_space.sample()
        else:
            agent, action = agent.sample(observation)

        # Execute action in environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        mask = 1.0 - float(terminated)
        if cfg.training.discount_grad:
            discount *= cfg.algorithm.discount

        # Store transition in replay buffer
        replay_buffer.add(observation, action, reward, mask, next_observation, discount)

        observation = next_observation

        # Reset environment if episode ends
        if done:
            observation, _ = env.reset()
            discount = 1.
            if cfg.logging.use_wandb and 'episode' in info:
                wandb.log({'train/episode_return': info['episode']['r'][0],
                          'train/episode_length': info['episode']['l'][0]},
                         step=step_num, commit=False)

        # Perform agent update if enough steps have passed
        if step_num >= cfg.training.start_steps:
            for i in range(cfg.training.update_frequency):
                batch = replay_buffer.sample(cfg.training.batch_size)
                agent, update_info = agent.update(batch)

            if cfg.logging.use_wandb and update_info:
                wandb.log({f'train/{k}': v for k, v in update_info.items()}, step=step_num, commit=False)

        # Evaluate agent periodically
        if step_num % cfg.training.eval_freq == 0:
            agent, avg_reward, avg_disc_reward = evaluate(agent, eval_env, cfg.training.eval_episodes, cfg.algorithm.discount)

            log_data = {"eval/avg_reward": avg_reward, "eval/avg_disc_reward": avg_disc_reward, "step": step_num}
            if cfg.logging.use_wandb:
                wandb.log(log_data, step=step_num, commit=True)

            print(f"---------------------------------------")
            print(f"Step: {step_num}, Evaluation Avg Reward: {avg_reward:.2f}, Evaluation Avg Discounted Reward: {avg_disc_reward:.2f}")
            print(f"---------------------------------------")
        
        # === NEW: Gradient Quality Evaluation for TD3-GC ===
        # Only evaluate gradient quality for TD3-GC agents and when replay buffer has enough data
        if (agent_name == "TD3GC" and 
            gradient_eval_env is not None and 
            step_num >= cfg.training.start_steps and 
            len(replay_buffer) >= 10000):  # Ensure we have enough data
            
            # Evaluate gradient quality every 50K steps
            gradient_eval_results = maybe_evaluate_gradient_quality(
                agent=agent,
                env=gradient_eval_env,
                replay_buffer=replay_buffer,
                step_num=step_num,
                evaluation_frequency=50_000,  # Every 50K steps
                rollout_steps=100_000,        # 100K rollout steps as requested
                critic_training_steps=50_000, # 50K training steps for fresh critic
                evaluation_batch_size=1000,   # Batch size for gradient comparison
                num_parallel_envs=8           # Use 8 parallel environments for faster rollouts
            )
            
            # The results are automatically logged to wandb within the function
            # but you can also access them here if needed for additional processing
            if gradient_eval_results:
                print(f"Completed gradient quality evaluation at step {step_num}")

    # Clean up
    env.close()
    eval_env.close()
    if gradient_eval_env is not None:
        gradient_eval_env.close()
    if cfg.logging.use_wandb:
        wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    main() 